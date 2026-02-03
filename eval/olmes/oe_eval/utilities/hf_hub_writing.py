import os

from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN", None))


def upload_to_hf(local_file_path, repo_id, repo_path, repo_type="dataset"):
    api.upload_file(
        path_or_fileobj=local_file_path,
        repo_id=repo_id,
        path_in_repo=repo_path,
        repo_type=repo_type,
    )
