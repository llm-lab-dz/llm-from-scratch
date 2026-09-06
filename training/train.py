"""
Training loop for the small GPT model, designed for a single Kaggle GPU
(T4 or P100). Saves checkpoints regularly so you can stop and resume
across Kaggle sessions.

Usage (on Kaggle, in a notebook cell):
    !python training/train.py --data_dir data --ckpt_dir /kaggle/working/checkpoints

To resume from a checkpoint:
    !python training/train.py --resume /kaggle/working/checkpoints/ckpt.pt
"""
import argparse
import os
import time
import numpy as np
import torch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model import GPT, GPTConfig


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=50):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_iters", type=int, default=6000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--ckpt_interval", type=int, default=250)
    parser.add_argument("--time_budget_min", type=float, default=55,
                         help="Stop training gracefully after this many minutes")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    config = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(config).to(device)
    print(f"Model has {model.num_params():,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    start_iter = 0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"] + 1

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    start_time = time.time()

    for it in range(start_iter, args.max_iters):
        elapsed_min = (time.time() - start_time) / 60
        if elapsed_min > args.time_budget_min:
            print(f"Time budget of {args.time_budget_min} min reached, stopping and saving.")
            break

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)

        with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.float16):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if it % 50 == 0:
            print(f"iter {it}: loss {loss.item():.4f} | elapsed {elapsed_min:.1f} min")

        if it % args.eval_interval == 0 and it > 0:
            losses = estimate_loss(model, train_data, val_data, args.block_size, args.batch_size, device)
            print(f"  -> eval: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if it % args.ckpt_interval == 0 and it > 0:
            ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "iter": it,
            }, ckpt_path)
            print(f"  -> saved checkpoint at iter {it} to {ckpt_path}")

    # Final save
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "iter": it,
    }, ckpt_path)
    print(f"Final checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()