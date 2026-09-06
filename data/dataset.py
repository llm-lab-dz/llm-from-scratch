"""
Downloads a small general-text + code + conversation mix, cleans it into
plain text files, then tokenizes everything into train.bin / val.bin
ready for training.

Run in two steps:
    python prepare_data.py --step download   # pulls raw text into data/raw/
    # then train your tokenizer (see tokenizer/train_tokenizer.py)
    python prepare_data.py --step tokenize --tokenizer ../tokenizer/tokenizer.json
"""
import argparse
import os
import urllib.request
import numpy as np

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    out = os.path.join(RAW_DIR, "shakespeare.txt")
    print(f"Downloading Tiny Shakespeare -> {out}")
    urllib.request.urlretrieve(url, out)


def download_codeparrot_slice(n_examples=3000):
    """Small slice of clean Python code from CodeParrot via Hugging Face datasets."""
    from datasets import load_dataset
    print("Streaming a slice of CodeParrot-clean...")
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
    out = os.path.join(RAW_DIR, "code.txt")
    with open(out, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            if i >= n_examples:
                break
            content = example.get("content", "")
            if content:
                f.write(content + "\n<|endoftext|>\n")
    print(f"Saved {n_examples} code files -> {out}")


def download_alpaca():
    """Instruction/response conversation data, formatted as chat turns."""
    from datasets import load_dataset
    print("Downloading Alpaca...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out = os.path.join(RAW_DIR, "conversations.txt")
    with open(out, "w", encoding="utf-8") as f:
        for ex in ds:
            instruction = ex["instruction"]
            inp = ex.get("input", "")
            output = ex["output"]
            prompt = instruction + ("\n" + inp if inp else "")
            f.write(f"<|user|>\n{prompt}\n<|assistant|>\n{output}\n<|endoftext|>\n")
    print(f"Saved conversations -> {out}")


def tokenize_and_save(tokenizer_path, val_fraction=0.05):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)

    all_ids = []
    for fname in sorted(os.listdir(RAW_DIR)):
        path = os.path.join(RAW_DIR, fname)
        print(f"Tokenizing {path} ...")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        ids = tok.encode(text).ids
        all_ids.extend(ids)
        print(f"  -> {len(ids):,} tokens")

    ids = np.array(all_ids, dtype=np.uint16)
    split = int(len(ids) * (1 - val_fraction))
    train_ids, val_ids = ids[:split], ids[split:]

    train_ids.tofile("data/train.bin")
    val_ids.tofile("data/val.bin")
    print(f"Total tokens: {len(ids):,} | train: {len(train_ids):,} | val: {len(val_ids):,}")
    print("Saved data/train.bin and data/val.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["download", "tokenize"], required=True)
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--code_examples", type=int, default=3000)
    args = parser.parse_args()

    if args.step == "download":
        download_shakespeare()
        download_codeparrot_slice(n_examples=args.code_examples)
        download_alpaca()
        print("\nDone. Raw text is in data/raw/. Now train the tokenizer, then re-run with --step tokenize.")
    else:
        tokenize_and_save(args.tokenizer)


if __name__ == "__main__":
    main()