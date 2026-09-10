import argparse
import os
import sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model import GPT


def sample(model, idx, max_new_tokens, block_size, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.3):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # repetition penalty: discourage tokens already generated
        for token_id in set(idx[0].tolist()):
            logits[0, token_id] /= repetition_penalty

        # top-k
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        # top-p (nucleus)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cum_probs > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        next_token_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_idx.gather(-1, next_token_sorted)

        idx = torch.cat([idx, next_token], dim=1)
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = Tokenizer.from_file(args.tokenizer)
    ids = tok.encode(args.prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = sample(model, idx, args.max_new_tokens, config.block_size,
                     temperature=args.temperature, top_k=args.top_k,
                     top_p=args.top_p, repetition_penalty=args.repetition_penalty)

    text = tok.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
