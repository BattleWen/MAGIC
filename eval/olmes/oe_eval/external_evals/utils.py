import logging
import subprocess

from oe_eval.utils import make_cli_command

logger = logging.getLogger()


def uv_run(cmd, args: dict, directory=None, run_prefix=""):
    uv_cmd = run_prefix + "uv run"
    if directory is not None:
        uv_cmd += f" --directory {directory}"
    full_cmd = make_cli_command(cmd, args)
    uv_cmd += f" {full_cmd}"
    logger.info(f"Running uv command: {uv_cmd}")
    return_code = subprocess.run(uv_cmd, shell=True).returncode
    return return_code
