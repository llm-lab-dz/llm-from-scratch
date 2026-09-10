"""
Downloads a ~1B-token mix of general web text, encyclopedic text, code, and
instruction data, then tokenizes it into train.bin / val.bin.

Data mix (defaults aim for ~1B tokens total; ~4 chars/token is used only as
a stopping heuristic while streaming -- the real count is printed at the end
of the tokenize step):

    OpenWebText   ~2.2GB text  -> ~550M tokens   (general web writing)
    Wikipedia     ~1.4GB text  -> ~350M tokens   (encyclopedic/factual text)
    CodeParrot    ~0.6GB text  -> ~150M tokens   (Python code)
    Alpaca        full dataset -> ~6M tokens     (instruction/response pairs)

All three big sources are streamed from Hugging Face (streaming=True), so
nothing downloads the full underlying dataset -- only as many characters as
you ask for come over the wire. Tokenization is also streamed chunk-by-chunk
straight into train.bin/val.bin, so RAM use stays flat no matter how large
the corpus gets. This matters a lot at this scale: holding ~1B token ids in
a plain Python list would need 25-30GB+ of RAM, which Kaggle doesn't have.

Run in three steps:
    python data/dataset.py --step download
    python tokenizer/tokenizer.py --files data/raw/*.txt --vocab_size 16000 --out tokenizer/tokenizer.json
    python data/dataset.py --step tokenize --tokenizer tokenizer/tokenizer.json --cleanup_raw
"""
import argparse
import json
import os
import urllib.request

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

GB = 1_000_000_000


def _stream_text_to_file(example_iter, text_getter, out_path, char_budget, print_every=200_000_000):
    """Stream examples from a HF `datasets` iterator into a plain text file
    until char_budget characters have been written. Never holds more than
    one example in memory at a time, so this is safe regardless of how large
    the underlying dataset actually is."""
    written = 0
    next_print = print_every
    with open(out_path, "w", encoding="utf-8") as f:
        for example in example_iter:
            text = text_getter(example)
            if not text:
                continue
            f.write(text)
            f.write("\n<|endoftext|>\n")
            written += len(text)
            if written >= next_print:
                print(f"  ... {written / 1e6:.0f}M / {char_budget / 1e6:.0f}M chars -> {out_path}")
                next_print += print_every
            if written >= char_budget:
                break
    print(f"Saved {written:,} characters -> {out_path}")
    return written


def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    out = os.path.join(RAW_DIR, "shakespeare.txt")
    print(f"Downloading Tiny Shakespeare -> {out}")
    urllib.request.urlretrieve(url, out)


def download_openwebtext(char_budget):
    from datasets import load_dataset
    print(f"Streaming OpenWebText (target ~{char_budget / GB:.1f}GB text)...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    out = os.path.join(RAW_DIR, "openwebtext.txt")
    _stream_text_to_file(ds, lambda ex: ex.get("text", ""), out, char_budget)


def download_wikipedia(char_budget):
    from datasets import load_dataset
    print(f"Streaming Wikipedia (target ~{char_budget / GB:.1f}GB text)...")
    # Pre-extracted parquet version -- no apache_beam build step needed.
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    out = os.path.join(RAW_DIR, "wikipedia.txt")
    _stream_text_to_file(ds, lambda ex: ex.get("text", ""), out, char_budget)


def download_codeparrot(char_budget):
    from datasets import load_dataset
    print(f"Streaming CodeParrot-clean (target ~{char_budget / GB:.1f}GB text)...")
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
    out = os.path.join(RAW_DIR, "code.txt")
    _stream_text_to_file(ds, lambda ex: ex.get("content", ""), out, char_budget)


def download_alpaca():
    """Small instruction/response dataset -- loaded in full, not streamed,
    since it's only ~52k short examples (tens of MB)."""
    from datasets import load_dataset
    print("Downloading Alpaca (full, ~52k examples)...")
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


def tokenize_and_save(tokenizer_path, val_fraction=0.05, chunk_chars=2_000_000, cleanup_raw=False):
    """Tokenizes every file in raw/ chunk-by-chunk, writing straight to
    train.bin/val.bin as it goes. Never accumulates the full token list in
    memory -- this is what makes tokenizing a multi-GB corpus safe on
    Kaggle's RAM budget. Each file's chunks are split train/val by chunk
    index (roughly val_fraction of chunks -> val), which needs no upfront
    knowledge of the file's total token count."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)

    train_path = os.path.join(SCRIPT_DIR, "train.bin")
    val_path = os.path.join(SCRIPT_DIR, "val.bin")
    val_every_n_chunks = max(1, round(1 / val_fraction))

    total_train, total_val = 0, 0
    raw_files = [
        fname for fname in sorted(os.listdir(RAW_DIR))
        if os.path.isfile(os.path.join(RAW_DIR, fname))
    ]
    with open(train_path, "wb") as train_f, open(val_path, "wb") as val_f:
        for fname in raw_files:
            path = os.path.join(RAW_DIR, fname)
            print(f"Tokenizing {path} ...")
            file_train, file_val = 0, 0
            chunk_idx = 0
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    text = f.read(chunk_chars)
                    if not text:
                        break
                    ids = tok.encode(text).ids
                    if ids and max(ids) >= np.iinfo(np.uint16).max:
                        raise ValueError("Token ID does not fit in uint16; use a larger dataset dtype")
                    arr = np.array(ids, dtype=np.uint16)
                    if chunk_idx % val_every_n_chunks == 0:
                        val_f.write(arr.tobytes())
                        file_val += len(arr)
                    else:
                        train_f.write(arr.tobytes())
                        file_train += len(arr)
                    chunk_idx += 1
            print(f"  -> {file_train + file_val:,} tokens ({file_train:,} train / {file_val:,} val)")
            total_train += file_train
            total_val += file_val

            if cleanup_raw:
                os.remove(path)
                print(f"  -> removed {path} to free disk space")

    print(f"\nTotal tokens: {total_train + total_val:,} | train: {total_train:,} | val: {total_val:,}")
    print(f"Saved {train_path} and {val_path}")
    metadata_path = os.path.join(SCRIPT_DIR, "dataset_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump({
            "tokenizer_path": os.path.abspath(tokenizer_path),
            "vocab_size": tok.get_vocab_size(),
            "dtype": "uint16",
            "val_fraction_requested": val_fraction,
            "chunk_chars": chunk_chars,
            "raw_files": raw_files,
            "train_tokens": total_train,
            "val_tokens": total_val,
        }, metadata_file, indent=2)
        metadata_file.write("\n")
    print(f"Saved {metadata_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["download", "tokenize"], required=True)
    parser.add_argument("--tokenizer", type=str,
                         default=os.path.join(SCRIPT_DIR, "..", "tokenizer", "tokenizer.json"))
    parser.add_argument("--openwebtext_gb", type=float, default=2.2,
                         help="GB of characters to stream from OpenWebText (~4 chars/token)")
    parser.add_argument("--wikipedia_gb", type=float, default=1.4)
    parser.add_argument("--code_gb", type=float, default=0.6)
    parser.add_argument("--skip_shakespeare", action="store_true")
    parser.add_argument("--cleanup_raw", action="store_true",
                         help="Delete raw/*.txt after tokenizing to free disk space, "
                              "since only train.bin/val.bin are needed for training")
    args = parser.parse_args()

    if args.step == "download":
        if not args.skip_shakespeare:
            download_shakespeare()
        download_openwebtext(args.openwebtext_gb * GB)
        download_wikipedia(args.wikipedia_gb * GB)
        download_codeparrot(args.code_gb * GB)
        download_alpaca()
        total_gb = args.openwebtext_gb + args.wikipedia_gb + args.code_gb
        print(f"\nDone. ~{total_gb:.1f}GB of raw text in {RAW_DIR} "
              f"(~{total_gb * 250:.0f}M tokens estimated).")
        print("Now train the tokenizer, then re-run with --step tokenize.")
    else:
        tokenize_and_save(args.tokenizer, cleanup_raw=args.cleanup_raw)


if __name__ == "__main__":
    main()