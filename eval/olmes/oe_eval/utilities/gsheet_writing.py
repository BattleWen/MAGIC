import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import pygsheets
from googleapiclient.errors import HttpError as GoogleApiClientHttpError

from oe_eval.utils import sanitize

logger = logging.getLogger(__name__)


# When we attempt to write results to a spreadsheet
# we'll use a worksheet within it as a kind of lock to avoid
# having multiple runs overwrite each other's results.
# This is the name of that worksheet.
LOCK_WORKSHEET_NAME = "lock"
# Sometimes, two different runs can try to create a
# 'lock worksheet' so close together, that we end up with
# two new worksheets. The spreadsheet will have one with
# GSHEET_LOCK_SHEET_NAME as its name, and other will
# start with this conflict prefix.
LOCK_WORKSHEET_CONFLICT_PREFIX = f"{LOCK_WORKSHEET_NAME}_conflict"

LOCK_TIME_CELL_ADDRESS = "A1"


@dataclass
class GSheetLoc:
    spreadsheet: str
    worksheet: str


class GSheetWriter:
    def __init__(
        self,
        service_account_json: Optional[str] = None,
        # Recommended not to change the default max_seconds_lock_can_be_held value.
        # Overriding the value is basically for tests.
        max_seconds_lock_can_be_held: Optional[float] = None,
    ):
        service_account_json = service_account_json or os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
        self.client = pygsheets.authorize(
            service_account_json=service_account_json,
            # check = False means pygsheets won't attempt to catch and sleep for rate limit errors
            # we're setting this to False so we can handle these ourselves and have more visibility
            # into what's happening
            check=False,
        )

        # fairly high expiration time for holding a lock to give a lock holder time
        # to do its thing (confirm they got the lock, write their results and release
        # the lock) in the case where we hit rate limits during that time
        self.max_seconds_lock_can_be_held = (
            timedelta(minutes=20).total_seconds()
            if max_seconds_lock_can_be_held is None
            else max_seconds_lock_can_be_held
        )

    @staticmethod
    def mk_conflict_worksheet_name(worksheet_id) -> str:
        return f"{LOCK_WORKSHEET_CONFLICT_PREFIX}{worksheet_id}"

    def _lock_is_expired(self, worksheet) -> bool:
        # Note: The caller is responsible for catching any additional exceptions.
        try:
            time_now = time.time()
            maybe_lock_time = worksheet.get_value(LOCK_TIME_CELL_ADDRESS)
            if maybe_lock_time == "":
                # The lock sheet doesn't have an entry indicating at what time
                # it was created. Assume it's not expired.
                return False
            else:
                lock_time = float(maybe_lock_time)
                seconds_lock_has_been_held = time_now - lock_time

                return seconds_lock_has_been_held > self.max_seconds_lock_can_be_held
        except GoogleApiClientHttpError as exc:
            if ("400" in str(exc)) and ("Unable to parse range" in str(exc)):
                logger.info(
                    "Unable to parse range error when checking if a lock is expired. Assuming it's not expired."
                )
                return False
            else:
                raise exc

    def _handle_lock_if_expired(self, spreadsheet_name: str):
        # Note: The caller is responsible for catching any exceptions

        logger.info("Checking if a lock is currently held and expired.")
        spreadsheet = self.client.open(spreadsheet_name)
        worksheets = spreadsheet.worksheets()
        worksheet_titles = [w.title for w in worksheets]
        if LOCK_WORKSHEET_NAME in worksheet_titles:
            lock_worksheet = spreadsheet.worksheet_by_title(LOCK_WORKSHEET_NAME)
            if self._lock_is_expired(lock_worksheet):
                logger.info(
                    "It appears that the lock currently held is expired. Deleting the lock worksheet."
                )
                self._release_lock(spreadsheet_name)

        logger.info("Done checking if a lock is currently held and expired.")

    def _maybe_get_lock_helper(self, spreadsheet_name: str):
        # Note: the caller is responsible for catching exceptions.
        logger.info("Attempting to create a lock worksheet.")

        spreadsheet_pre_creation = self.client.open(spreadsheet_name)
        created_worksheet = spreadsheet_pre_creation.add_worksheet(title=LOCK_WORKSHEET_NAME)
        # So we know if we hold the lock too long.
        # Do this now, even if we didn't actually get the lock for sure.
        created_worksheet.update_value(LOCK_TIME_CELL_ADDRESS, time.time())
        created_worksheet_id = created_worksheet.id
        logger.info(f"Maybe got lock. Worksheet id: {created_worksheet_id}.")

        # Things seem to work without a sleep too, but seems possibly safer to
        # leave this in.
        time.sleep(1)
        logger.info("Done attempting to create a lock worksheet.")

        return created_worksheet_id

    def _confirm_lock_helper(self, spreadsheet_name: str, created_worksheet_id: int) -> bool:
        # Note: the caller is responsible for catching exceptions.
        logger.info("Attempting to confirm we got the lock.")

        # Note: it is important to open the spreadsheet again.
        spreadsheet_post_creation = self.client.open(spreadsheet_name)
        worksheets = spreadsheet_post_creation.worksheets()

        # Check with both title and id so that we find all the worksheets
        # we might care about.
        worksheets_with_the_right_title = [w for w in worksheets if w.title == LOCK_WORKSHEET_NAME]
        worksheets_with_the_right_id = [w for w in worksheets if w.id == created_worksheet_id]
        num_worksheets_with_the_right_title = len(worksheets_with_the_right_title)
        num_worksheets_with_the_right_id = len(worksheets_with_the_right_id)

        # Not all these cases are fully understood,
        # but we'll try to log enough to understand them better...
        # Get through the unexpected ones first.
        if num_worksheets_with_the_right_id == 0:
            # I don't think this can actually happen.
            # But let's log just in case.
            to_log = (
                f"This is unexpected. We have no worksheets with id {created_worksheet_id}. "
                f"Tell chloea."
            )
            logger.info(to_log)
            # Let the wrapper _try_to_get_lock() deal with it.
            raise Exception(to_log)

        elif num_worksheets_with_the_right_title > 1:
            # I don't think this can actually happen.
            # But let's log just in case.
            to_log = (
                f"This is unexpected. We shouldn't have more than one worksheet with the same title, "
                f"but we have {num_worksheets_with_the_right_title}. Tell chloea."
            )
            logger.info(to_log)
            # Let the wrapper _try_to_get_lock() deal with it.
            raise Exception(to_log)

        elif num_worksheets_with_the_right_id > 1:
            # I don't think this can actually happen.
            # But let's log just in case.
            to_log = (
                f"This is unexpected. We shouldn't have more than one worksheet with the same id, "
                f"but we have {num_worksheets_with_the_right_id}. Tell chloea."
            )
            logger.info(to_log)
            # Let the wrapper _try_to_get_lock() deal with it.
            raise Exception(to_log)

        else:
            # At this point there's basically three cases:
            # - no one has the lock (also have a conflict worksheet in this case)
            # - someone else has the lock (also have a conflict worksheet in this case)
            # - we have the lock

            # This should be true given the checks above.
            assert num_worksheets_with_the_right_id == 1
            logger.info("We have one worksheet with the right id.")
            [worksheet_with_the_right_id] = worksheets_with_the_right_id

            if num_worksheets_with_the_right_title == 0:
                # Neither we, nor someone else, has the lock right now.
                # I think this could happen if we and someone else tried to get the lock
                # at the same time, someone else got it, wrote what they wanted, and
                # released it before we got to the point above where we look up worksheets
                # by title.
                logger.info(f"There is no worksheet called {LOCK_WORKSHEET_NAME} right now.")
                logger.info("No one has the lock.")
                got_lock = False
                conflict_case = True

            else:
                # This should be true given the checks above.
                assert num_worksheets_with_the_right_title == 1
                logger.info("We have one worksheet with the right title.")
                [worksheet_with_the_right_title] = worksheets_with_the_right_title

                # We have one worksheet with the right title,
                # and one worksheet with the right id (they may or may not be the same worksheet).

                # it's possible we have the lock, but it's also possible someone else does

                # Both ids and titles should match, or neither should match.
                # If we're in a different case, we don't know how to deal with that.
                # But we'll log just in case.
                titles_match = worksheet_with_the_right_id.title == LOCK_WORKSHEET_NAME
                ids_match = worksheet_with_the_right_title.id == created_worksheet_id
                logger.info(f"titles match: {titles_match}, ids match: {ids_match}")
                if titles_match != ids_match:
                    logger.info("Unexpected state. Tell chloea.")

                [things_match] = list(set([titles_match, ids_match]))
                if things_match:
                    # We're the ones that created the current lock worksheet.
                    # That means we 'got the lock'.
                    logger.info("Confirmed that we got the lock.")
                    got_lock = True
                    conflict_case = False

                else:
                    # We tried to get a lock/create a lock worksheet at the same time as someone else.
                    # The worksheet we created ended up being the one in conflict, so someone else
                    # 'has the lock'.
                    logger.info("Someone else has the lock.")
                    got_lock = False
                    conflict_case = True

            if conflict_case:
                # Delete the conflict worksheet we created.
                assert worksheet_with_the_right_id.title == GSheetWriter.mk_conflict_worksheet_name(
                    created_worksheet_id
                )
                logger.info(
                    f"Deleting conflict worksheet with name {worksheet_with_the_right_id.title} and id {worksheet_with_the_right_id.id}."
                )
                spreadsheet_post_creation.del_worksheet(worksheet_with_the_right_id)

            return got_lock

    def _get_lock_helper(self, spreadsheet_name: str) -> bool:
        # Note: the caller is responsible for catching exceptions.
        # Expects that it will be called by _try_to_get_lock() via _try_to_get_lock_helper().

        logger.info(f"Attempting to get lock for spreadsheet with title: {spreadsheet_name}.")
        try:
            # Some potential clean up to start.
            GSheetWriter._sleep_on_rate_limit(
                lambda: self._handle_lock_if_expired(spreadsheet_name)
            )

            # Make a lock worksheet.
            # Don't wrap this bit in _retry_activity_on_rate_limit() - if we hit a rate limit here
            # that's fine. Next steps will get handled elsewhere (in _try_to_get_lock()).
            created_worksheet_id = GSheetWriter._sleep_on_rate_limit(
                lambda: self._maybe_get_lock_helper(spreadsheet_name)
            )

            # Now let's try to figure out if anyone else was trying to do the same thing at
            # the same time.
            # Wrap it in _retry_activity_on_rate_limit() so we can still attempt to confirm
            # even if we're being initially rate-limited. Helps avoid cases where someone
            # gets the lock, but doesn't realize it, and so doesn't release the lock, meaning
            # no one else can get the lock until it expires.
            return self._retry_activity_on_rate_limit(
                lambda: self._confirm_lock_helper(spreadsheet_name, created_worksheet_id)
            )

        except GoogleApiClientHttpError as exc:
            if f'A sheet with the name "{LOCK_WORKSHEET_NAME}" already exists' in str(exc):
                # Someone else already created a lock worksheet. They 'have the lock'.
                logger.info("Someone else has the lock.")
                return False
            elif GSheetWriter._exception_looks_like_rate_limit(exc):
                # If we hit this rate limit here, sleep for an extra minute,
                # to give whoever does have the lock space to make their updates
                # and release the lock (if someone has the lock).
                sleep_for = 60
                logger.info(
                    f"Sleeping for another {sleep_for} seconds to give the potential lock holder space to not bump into rate limits."
                )
                time.sleep(sleep_for)
                # _try_to_get_lock() will also add some jitter
                return False
            else:
                # Anything else is unexpected. Let the caller deal with it.
                raise exc

    def _try_to_get_lock_helper(self, spreadsheet_name: str) -> bool:
        return GSheetWriter._try_activity_and_catch_exceptions(
            lambda: self._get_lock_helper(spreadsheet_name=spreadsheet_name),
            default_return=False,
        )

    def _try_to_get_lock(
        self, spreadsheet_name: str, attempts_remaining: int = 30, max_extra_sleep_secs: int = 60
    ) -> bool:
        if attempts_remaining > 0:
            got_lock = self._try_to_get_lock_helper(spreadsheet_name)
            if got_lock:
                return True
            else:
                # someone else is trying to write to the same file
                # sleep and try again in a bit
                to_sleep = 5 + random.randint(1, max_extra_sleep_secs)
                new_attempts_remaining = attempts_remaining - 1
                logger.info(
                    f"Failed to get the lock this attempt. Sleeping for {to_sleep} seconds."
                )
                time.sleep(to_sleep)
                logger.info(f"Done sleeping. Attempts left: {new_attempts_remaining}.")
                return self._try_to_get_lock(
                    spreadsheet_name=spreadsheet_name,
                    attempts_remaining=new_attempts_remaining,
                    max_extra_sleep_secs=max_extra_sleep_secs,
                )
        else:
            logger.info(
                f"We've exhausted the allowed attempts for getting the lock to write to spreadsheet {spreadsheet_name}."
            )
            return False

    def _release_lock(self, spreadsheet_name: str) -> bool:
        # Note: the caller is responsible for catching exceptions.

        logger.info(f"Attempting to release lock for spreadsheet with title: {spreadsheet_name}.")
        spreadsheet = self.client.open(spreadsheet_name)
        worksheet = spreadsheet.worksheet_by_title(LOCK_WORKSHEET_NAME)
        spreadsheet.del_worksheet(worksheet)
        logger.info("Released lock.")
        return True

    def _try_to_release_lock(self, spreadsheet_name: str):
        return GSheetWriter._try_activity_and_catch_exceptions(
            lambda: self._retry_activity_on_rate_limit(
                lambda: self._release_lock(spreadsheet_name=spreadsheet_name)
            ),
            default_return=False,
        )

    def _update_spreadsheet(
        self, spreadsheet_name: str, rows: List[Dict], worksheet_title: str = "Sheet1"
    ) -> GSheetLoc:
        # Note: the caller is responsible for catching exceptions.
        # Prefer not to call this directly. Use _try_to_write_results() instead.

        logger.info(f"Updating worksheet {worksheet_title}.")
        new_df = pd.DataFrame(rows)

        spreadsheet = self.client.open(spreadsheet_name)

        # make worksheet if doesn't exist
        if worksheet_title in [s.title for s in spreadsheet.worksheets()]:
            worksheet = spreadsheet.worksheet_by_title(worksheet_title)
        else:
            spreadsheet.add_worksheet(
                rows=new_df.shape[0], cols=new_df.shape[1], title=worksheet_title
            )
            worksheet = spreadsheet.worksheet_by_title(worksheet_title)
        current_df = worksheet.get_as_df()
        new_df = pd.concat([current_df, new_df])
        worksheet.set_dataframe(new_df, (1, 1), nan="")
        logger.info(f"Done updating worksheet {worksheet_title}.")

        return GSheetLoc(
            spreadsheet=spreadsheet_name,
            worksheet=worksheet_title,
        )

    def _try_to_update_spreadsheet(
        self, spreadsheet_name: str, rows: List[Dict], worksheet_title: str = "Sheet1"
    ) -> Optional[GSheetLoc]:
        return GSheetWriter._try_activity_and_catch_exceptions(
            lambda: self._retry_activity_on_rate_limit(
                lambda: self._update_spreadsheet(
                    spreadsheet_name=spreadsheet_name, rows=rows, worksheet_title=worksheet_title
                )
            ),
            default_return=None,
        )

    def _try_lockless_attempt_to_save_somewhere(
        self, spreadsheet_name: str, rows: List[Dict]
    ) -> Optional[GSheetLoc]:
        worksheet_title = f"extra-{time.time()}_{random.randint(0, 1000)}"
        return self._try_to_update_spreadsheet(
            spreadsheet_name=spreadsheet_name, rows=rows, worksheet_title=worksheet_title
        )

    def try_to_write_results(
        self, spreadsheet_name: str, rows: List[Dict], worksheet_title: str = "Sheet1"
    ) -> Optional[GSheetLoc]:
        def log_results():
            logger.info("Results we would have written:")
            for r in rows:
                logger.info(r)
            logger.info("Done logging results we would have written.")

        if (worksheet_title == LOCK_WORKSHEET_NAME) or (
            worksheet_title.startswith(LOCK_WORKSHEET_CONFLICT_PREFIX)
        ):
            logger.info(
                f"Cannot have a worksheet with name {worksheet_title}. It clashes with our lock approach. Not writing results."
            )
            outcome = None

        else:
            if self._try_to_get_lock(spreadsheet_name=spreadsheet_name):
                outcome = self._try_to_update_spreadsheet(
                    spreadsheet_name=spreadsheet_name,
                    rows=rows,
                    worksheet_title=worksheet_title,
                )
                self._try_to_release_lock(spreadsheet_name=spreadsheet_name)
            else:
                logger.info(
                    f"Was unable to get the lock for spreadsheet {spreadsheet_name}. Not writing results to worksheet {worksheet_title}."
                )
                logger.info(
                    f"Attempting to write results to a different worksheet instead of {worksheet_title}."
                )
                outcome = self._try_lockless_attempt_to_save_somewhere(
                    spreadsheet_name=spreadsheet_name, rows=rows
                )
                if outcome is not None:
                    logger.info(f"See worksheet {outcome.worksheet}.")

        if outcome is None:
            logger.info("Looks like the write attempt was unsuccessful.")
            log_results()

        return outcome

    @staticmethod
    def _try_activity_and_catch_exceptions(activity: Callable, default_return):
        # We usually want to be really generous about catching exceptions, so
        # that a job doesn't end up failing because of an unexpected issue
        # with this bit. This is a helper around that.
        try:
            outcome = activity()
            return outcome
        except Exception as exc:
            logger.info(f"Something unexpected went wrong doing this activity. Details: {exc}")
            return default_return

    @staticmethod
    def _exception_looks_like_rate_limit(exc: Exception):
        return "429" in str(exc) or "RATE_LIMIT_EXCEEDED" in str(exc)

    @staticmethod
    def _sleep_on_rate_limit(activity: Callable):
        try:
            outcome = activity()
            return outcome
        except GoogleApiClientHttpError as exc:
            if GSheetWriter._exception_looks_like_rate_limit(exc):
                sleep_for = 60
                logger.info(
                    f"Looks like we've been making too many requests to the google sheets API. Sleep for {sleep_for} seconds."
                )
                time.sleep(sleep_for)

            # callers might want to do more with this so still throw it
            # all we're doing here is making sure we sleep if we get rate limited
            raise exc

    @staticmethod
    def _retry_activity_on_rate_limit(
        activity: Callable,
        attempts_remaining: int = 3,
    ):
        # We're only using this in certain cases where it's relatively important that the activity eventually succeeds.
        # In other cases where we'll get rate limited, we'll say whatever we were trying to do failed, and move on.

        # The caller is responsible for catching this.
        # We want to make it obvious when we're in this case vs a case where activity is successful but returns None.
        assert attempts_remaining > 0, "We've exhausted our attempts for this activity."

        try:
            outcome = GSheetWriter._sleep_on_rate_limit(lambda: activity())
            return outcome
        except GoogleApiClientHttpError as exc:
            if GSheetWriter._exception_looks_like_rate_limit(exc):
                # We're not done in this case
                new_attempts_remaining = attempts_remaining - 1
                logger.info(
                    f"Retry the rate-limited activity. Attempts remaining: {new_attempts_remaining}."
                )
                return GSheetWriter._retry_activity_on_rate_limit(
                    activity=activity,
                    attempts_remaining=new_attempts_remaining,
                )
            else:
                # Something else went wrong. We don't know how to handle that.
                raise exc


def write_to_gsheet(
    gsheet: str,
    rows: List[Dict],
    sheet_title: str = "Sheet1",
    service_account_json: Optional[str] = None,
) -> Optional[GSheetLoc]:
    return GSheetWriter(service_account_json).try_to_write_results(
        spreadsheet_name=gsheet,
        rows=rows,
        worksheet_title=sheet_title,
    )


def convert_metrics_to_gsheet(metrics: dict, task_defaults: dict) -> dict:
    res: dict = {}
    model_config = metrics.get("model_config", {})
    task_config = metrics.get("task_config", {})
    all_metrics = metrics["metrics"].copy()
    all_metrics.pop("task", None)
    all_metrics.pop("num_instances", None)
    res["date"] = metrics.get("current_date", "")
    res["model"] = model_config.get("model", "")
    res["model_hash"] = metrics.get("model_hash", "")
    res["model_config"] = json.dumps(model_config)
    res["task"] = metrics.get("task_name", "")
    res["task_hash"] = metrics.get("task_hash", "")
    res["task_config"] = json.dumps(task_config)
    res["primary_metric"] = task_config.get("primary_metric", "")
    res["primary_score"] = metrics["metrics"].get("primary_score", "")
    res["processing_time"] = metrics.get("processing_time", "")
    res["num_instances"] = metrics.get("num_instances", "")
    res["limit"] = task_config.get("limit", task_defaults["limit"])
    res["split"] = task_config.get("split", task_defaults["split"])
    res["num_shots"] = task_config.get("num_shots", task_defaults["num_shots"])
    res["random_subsample_seed"] = task_config.get(
        "random_subsample_seed", task_defaults["random_subsample_seed"]
    )
    res["all_metrics"] = json.dumps(all_metrics)
    res["beaker_experiment_id"] = metrics.get("beaker_info", {}).get("BEAKER_EXPERIMENT_ID", "")
    res["compute_config"] = json.dumps(metrics["compute_config"])
    res = sanitize(res)
    return res


def write_metrics_to_gsheet(
    metrics: Union[List[dict], dict],
    gsheet: str,
    task_defaults: dict,
    sheet_title="Sheet1",
    service_account_json: Optional[str] = None,
):
    metrics_list = metrics if not isinstance(metrics, dict) else [metrics]
    res = [convert_metrics_to_gsheet(metrics1, task_defaults) for metrics1 in metrics_list]
    try:
        write_to_gsheet(
            gsheet, res, sheet_title=sheet_title, service_account_json=service_account_json
        )
    except Exception as e:
        logger.warning(f"Something went wrong when writing Google Sheet: {e}")

    if isinstance(metrics, dict):
        return res[0]
    return res
