"""
Push/pull processed training data (train.bin, val.bin, tokenizer.json) to a
Hugging Face Hub *dataset* repo, so the expensive one-time download+tokenize
step doesn't need to be repeated in every Kaggle training session.

This is separate from training/hf_checkpoint.py, which pushes/pulls model
checkpoints to a *model* repo -- use two different repo names, e.g.
  yourname/mini-gpt-data   (dataset repo, this file)
  yourname/mini-gpt-100m   (model repo, hf_checkpoint.py)

CLI usage:
    python hf_data.py --action push --repo_id yourname/mini-gpt-data --local_dir data
    python hf_data.py --action pull --repo_id yourname/mini-gpt-data --local_dir data
"""
import os
import shutil
from huggingface_hub import HfApi, hf_hub_download, create_repo


def ensure_dataset_repo(repo_id: str, token: str = None, private: bool = True):
    token = token or os.environ.get("HF_TOKEN")
    create_repo(repo_id, token=token, private=private, repo_type="dataset", exist_ok=True)


def push_data_files(local_dir: str, repo_id: str, filenames, token: str = None):
    """Upload each file in `filenames` (looked up as local_dir/filename) to
    the dataset repo, at the same filename in the repo root."""
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi()
    for fname in filenames:
        local_path = os.path.join(local_dir, fname)
        if not os.path.exists(local_path):
            print(f"Skipping {fname}: not found at {local_path}")
            continue
        print(f"Uploading {fname} ({os.path.getsize(local_path) / 1e6:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"add/update {fname}",
        )
    print(f"Done pushing to {repo_id}")


def pull_data_files(repo_id: str, local_dir: str, filenames, token: str = None):
    """Download each file in `filenames` from the dataset repo into local_dir.
    Returns the list of filenames that were successfully downloaded (missing
    files, e.g. on a first-ever run, are skipped rather than raising)."""
    token = token or os.environ.get("HF_TOKEN")
    os.makedirs(local_dir, exist_ok=True)
    downloaded = []
    for fname in filenames:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset", token=token)
        except Exception as e:
            print(f"Could not download {fname} from {repo_id} ({e.__class__.__name__}: {e})")
            continue
        local_path = os.path.join(local_dir, fname)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        shutil.copy(path, local_path)
        print(f"Downloaded {fname} -> {local_path}")
        downloaded.append(fname)
    return downloaded


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["push", "pull"], required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--local_dir", type=str, default=".")
    parser.add_argument("--files", nargs="+", default=[
        "train.bin", "val.bin", "tokenizer.json", "dataset_metadata.json"
    ])
    args = parser.parse_args()

    if args.action == "push":
        ensure_dataset_repo(args.repo_id)
        push_data_files(args.local_dir, args.repo_id, args.files)
    else:
        pull_data_files(args.repo_id, args.local_dir, args.files)