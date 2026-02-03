import logging
import os
import subprocess
from typing import List
from urllib.parse import urlparse

import boto3
import smart_open
from tqdm import tqdm

logger = logging.getLogger(__name__)

S3_CACHE_DIR = os.environ.get("S3_CACHE_DIR") or os.path.expanduser("~/.cache/oe_eval_s3_cache")
GS_CACHE_DIR = os.environ.get("GS_CACHE_DIR") or os.path.expanduser("~/.cache/oe_eval_gs_cache")


def cache_s3_folder(s3_path, cache_dir=None):
    cache_dir = cache_dir or S3_CACHE_DIR
    parsed_url = urlparse(s3_path)
    bucket_name = parsed_url.netloc
    s3_folder = parsed_url.path.lstrip("/")
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    files: List[str] = []
    s3_filenames = []
    for obj in bucket.objects.filter(Prefix=s3_folder):
        s3_filenames.append(obj.key)
    logger.info(f"Downloading {len(s3_filenames)} files from {s3_path} to {cache_dir}")
    for s3_filename in s3_filenames:
        local_filename = os.path.join(cache_dir, s3_filename)
        files.append(local_filename)
        if os.path.exists(local_filename):
            continue
        os.makedirs(os.path.dirname(local_filename), exist_ok=True)
        bucket.download_file(s3_filename, local_filename)
    local_dir = sorted([os.path.dirname(file) for file in files], key=len)[0]
    logger.info(f"Finished downloading to {local_dir}")
    return {"local_dir": local_dir, "files": files}


def cache_gs_folder(
    gs_path, cache_dir=None, parallel_thread_count=1, sliced_download_threshold="150M"
):
    """
    Cache a Google Cloud Storage folder using gsutil CLI.
    @vwxyzjn: setting `parallel_thread_count=1` is important for downloading large files from GS.

    Args:
        gs_path: Path to the GCS folder (gs://bucket-name/folder)
        cache_dir: Local directory to cache the files (defaults to GS_CACHE_DIR)
        parallel_thread_count: Number of threads for parallel download
        sliced_download_threshold: Threshold for sliced downloads

    Returns:
        dict: Contains local_dir and list of downloaded files
    """
    cache_dir = cache_dir or GS_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    parsed_url = urlparse(gs_path)
    bucket_name = parsed_url.netloc
    gs_folder = parsed_url.path.lstrip("/")

    # Determine the local path
    path_parts = gs_folder.rstrip("/").split("/")
    local_dir = os.path.join(cache_dir, bucket_name, *path_parts)
    local_dir_without_last_part = os.path.join(cache_dir, bucket_name, *path_parts[:-1])
    os.makedirs(local_dir_without_last_part, exist_ok=True)

    logger.info(f"Downloading files from {gs_path} to {local_dir}")

    # Construct the gsutil command
    cmd = [
        "gsutil",
        "-o",
        f"GSUtil:parallel_thread_count={parallel_thread_count}",
        "-o",
        f"GSUtil:sliced_object_download_threshold={sliced_download_threshold}",
        "-m",
        "cp",
        "-r",
        gs_path,
        local_dir_without_last_part,
    ]
    logger.info("running command: " + " ".join(cmd))

    # Execute the command and capture output
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    # Display output in real-time
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            if line.strip():
                logger.info(line.strip())
    else:
        logger.warning("No stdout available from gsutil command")

    process.wait()

    if process.returncode != 0:
        # Get the actual error message from the process
        process_error = process.stderr.read() if process.stderr else "No error message available"
        error_message = (
            f"gsutil command failed with return code {process.returncode}: {process_error}"
        )
        logger.error(error_message)
        raise Exception(error_message)

    # Get list of downloaded files
    files = []
    for root, _, filenames in os.walk(local_dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))

    logger.info(f"Finished downloading {len(files)} files to {local_dir}")
    return {"local_dir": local_dir, "files": files}


def upload_directory(local_dir: str, remote_dir: str):
    local_paths = [
        os.path.join(root, post_fn) for root, _, files in os.walk(local_dir) for post_fn in files
    ]
    dest_paths = [
        f"{remote_dir.rstrip('/')}/{os.path.relpath(local_path, local_dir).lstrip('/')}"
        for local_path in local_paths
    ]
    it = tqdm(
        iterable=zip(local_paths, dest_paths),
        total=len(local_paths),
        desc=f"Uploading files from {local_dir} to {remote_dir}",
    )
    for local_path, dest_path in it:
        with smart_open.open(local_path, "rb") as f, smart_open.open(dest_path, "wb") as g:
            g.write(f.read())
