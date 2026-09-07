"""
Push/pull training checkpoints to a Hugging Face Hub model repo, so training
survives across separate Kaggle sessions (Kaggle caps a "Save & Run All" at
12 hours per session).

Setup (once): create a HF access token (write access) and add it as a Kaggle
secret named HF_TOKEN. Then at the top of your notebook:

    from kaggle_secrets import UserSecretsClient
    import os
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

train.py calls these automatically when you pass --hf_repo yourname/your-model.
"""
import os
import shutil
from huggingface_hub import HfApi, hf_hub_download, create_repo


def ensure_repo(repo_id: str, token: str = None, private: bool = True):
    """Create the HF model repo if it doesn't exist yet. Safe to call every run."""
    token = token or os.environ.get("HF_TOKEN")
    create_repo(repo_id, token=token, private=private, repo_type="model", exist_ok=True)


def push_checkpoint(local_path: str, repo_id: str, path_in_repo: str = "ckpt.pt", token: str = None):
    """Upload a checkpoint file to the HF repo, overwriting the previous one at
    the same path so the repo always holds the latest checkpoint."""
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message=f"checkpoint update: {path_in_repo}",
    )


def pull_checkpoint(repo_id: str, local_dir: str, path_in_repo: str = "ckpt.pt", token: str = None):
    """Download the checkpoint from the HF repo if it exists.

    Returns the local path, or None if the repo/file doesn't exist yet (i.e.
    this is the very first run) so the caller can fall back to training from
    scratch.
    """
    token = token or os.environ.get("HF_TOKEN")
    os.makedirs(local_dir, exist_ok=True)
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="model",
            token=token,
        )
    except Exception as e:
        print(f"No existing checkpoint found on {repo_id} ({e.__class__.__name__}: {e}). "
              f"Starting fresh.")
        return None

    local_path = os.path.join(local_dir, os.path.basename(path_in_repo))
    shutil.copy(downloaded_path, local_path)
    print(f"Downloaded checkpoint from {repo_id}/{path_in_repo} -> {local_path}")
    return local_path