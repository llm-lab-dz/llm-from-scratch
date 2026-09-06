"""
Loads a trained checkpoint and generates text from a prompt.

Usage:
    python generate.py --ckpt /kaggle/working/checkpoints/ckpt.pt \
        --tokenizer tokenizer/tokenizer.json \
        --prompt "Once upon a time" --max_new_tokens 200
"""
import argparse
import os
import sys
import torch
from tokenizers import Tokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model import GPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]

    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = Tokenizer.from_file(args.tokenizer)
    ids = tok.encode(args.prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    text = tok.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()