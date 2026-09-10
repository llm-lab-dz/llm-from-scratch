"""
Trains a Byte-Pair Encoding (BPE) tokenizer on your raw text corpus.
Run this AFTER you've downloaded your raw text files with data/dataset.py's
download step (or point --files at any .txt files you already have).

Usage:
    python tokenizer/tokenizer.py --files data/raw/*.txt --vocab_size 16000 --out tokenizer/tokenizer.json
"""
import argparse
import glob
from tokenizers import ByteLevelBPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True,
                         help="Glob pattern(s) for raw text files, e.g. data/raw/*.txt")
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--out", type=str, default="tokenizer.json")
    args = parser.parse_args()

    # Expand any glob patterns passed in
    file_list = []
    for pattern in args.files:
        file_list.extend(glob.glob(pattern))
    if not file_list:
        raise ValueError(f"No files matched: {args.files}")

    print(f"Training BPE tokenizer on {len(file_list)} file(s): {file_list}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=file_list,
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|user|>", "<|assistant|>"],
    )

    tokenizer.save(args.out)
    print(f"Saved tokenizer to {args.out}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()