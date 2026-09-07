"""
Training loop for the ~100M-parameter GPT model. Runs on Kaggle's 2xT4 (via
DDP) or a single P100/T4, and pushes checkpoints to a Hugging Face Hub model
repo so training survives across Kaggle's 12-hour session cap.

--- Single GPU (P100, or one T4) ---
!python training/train.py --data_dir data --ckpt_dir /kaggle/working/checkpoints \
    --hf_repo yourname/your-model-name

--- Two T4s (DDP) ---
!torchrun --standalone --nproc_per_node=2 training/train.py --data_dir data \
    --ckpt_dir /kaggle/working/checkpoints --hf_repo yourname/your-model-name

Resuming is automatic: at startup the script tries to pull the latest
checkpoint from --hf_repo, falls back to a local --resume path if given, and
otherwise starts from scratch. It also saves + pushes a checkpoint whenever
the loop exits for any reason (time budget reached, finished, crashed,
Ctrl-C), so a session dying mid-run costs you at most --ckpt_interval steps.
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model import GPT, GPTConfig
from training.hf_checkpoint import ensure_repo, push_checkpoint, pull_checkpoint


def setup_distributed():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    is_master = rank == 0
    return ddp, rank, local_rank, world_size, device, is_master


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if "cuda" in str(device):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=50):
    model.eval()
    out = {}
    use_cuda = "cuda" in str(device)
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            with torch.autocast(device_type="cuda" if use_cuda else "cpu",
                                 dtype=torch.float16, enabled=use_cuda):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it, warmup_iters, lr_decay_iters, lr, min_lr):
    if it < warmup_iters:
        return lr * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def save_checkpoint(path, raw_model, optimizer, scaler, config, it, args):
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "iter": it,
        "args": vars(args),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                         help="Local checkpoint path to resume from if no HF checkpoint is found")
    parser.add_argument("--hf_repo", type=str, default=None,
                         help="e.g. 'yourname/your-model' -- if set, checkpoints are pulled/pushed here")
    parser.add_argument("--hf_ckpt_name", type=str, default="ckpt.pt")

    # model
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_checkpoint", action="store_true",
                         help="Trade compute for memory -- turn on if you hit CUDA OOM")

    # optimization
    parser.add_argument("--batch_size", type=int, default=16, help="micro-batch size, per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--max_iters", type=int, default=20000)
    parser.add_argument("--lr_decay_iters", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # logging / checkpointing cadence
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--ckpt_interval", type=int, default=200,
                         help="Steps between checkpoint pushes -- kept short so a crashed "
                              "session never loses much progress")
    parser.add_argument("--time_budget_min", type=float, default=690,
                         help="Stop gracefully after this many minutes (leaves buffer inside "
                              "Kaggle's 12h/720min cap for the final save+upload)")
    args = parser.parse_args()

    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.max_iters

    ddp, rank, local_rank, world_size, device, is_master = setup_distributed()
    torch.manual_seed(1337 + rank)
    np.random.seed(1337 + rank)
    random.seed(1337 + rank)

    if "cuda" in str(device):
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.ckpt_dir, exist_ok=True)
    local_ckpt_path = os.path.join(args.ckpt_dir, args.hf_ckpt_name)

    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    config = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    model.use_grad_checkpoint = args.grad_checkpoint
    if is_master:
        print(f"Model has {model.num_params():,} parameters")
        eff_batch = args.batch_size * args.grad_accum_steps * world_size
        print(f"Effective batch: {args.batch_size} x {args.grad_accum_steps} accum x "
              f"{world_size} GPU(s) = {eff_batch} sequences/step "
              f"(~{eff_batch * args.block_size:,} tokens/step)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                   betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=("cuda" in str(device)))
    start_iter = 0

    # --- resume: try HF Hub first, then a local --resume path, then scratch ---
    ckpt_to_load = None
    if args.hf_repo:
        if is_master:
            ensure_repo(args.hf_repo)
            ckpt_to_load = pull_checkpoint(args.hf_repo, args.ckpt_dir, args.hf_ckpt_name)
        if ddp:
            # Broadcast rank0's decision so every rank agrees on whether a
            # checkpoint was actually found -- don't let other ranks guess
            # based on stale local files.
            obj_list = [ckpt_to_load] if is_master else [None]
            dist.broadcast_object_list(obj_list, src=0)
            ckpt_to_load = obj_list[0]
    if ckpt_to_load is None and args.resume and os.path.exists(args.resume):
        ckpt_to_load = args.resume

    if ckpt_to_load:
        if is_master:
            print(f"Resuming from {ckpt_to_load}")
        ckpt = torch.load(ckpt_to_load, map_location=device, weights_only=False)        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
            except Exception:
                pass  # rng state from a different GPU count/config -- safe to skip
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])
        start_iter = ckpt["iter"] + 1
    else:
        if is_master:
            print("No checkpoint found -- starting from scratch")

    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module

    start_time = time.time()
    it = start_iter
    try:
        for it in range(start_iter, args.max_iters):
            elapsed_min = (time.time() - start_time) / 60
            if elapsed_min > args.time_budget_min:
                if is_master:
                    print(f"Time budget of {args.time_budget_min} min reached, stopping and saving.")
                break

            lr = get_lr(it, args.warmup_iters, args.lr_decay_iters, args.lr, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            for micro_step in range(args.grad_accum_steps):
                x, y = get_batch(train_data, args.block_size, args.batch_size, device)
                if ddp:
                    # only all-reduce gradients on the last micro-step of the window
                    model.require_backward_grad_sync = (micro_step == args.grad_accum_steps - 1)
                with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu",
                                     dtype=torch.float16, enabled=("cuda" in str(device))):
                    _, loss = model(x, y)
                    loss = loss / args.grad_accum_steps
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if is_master and it % args.log_interval == 0:
                tokens_per_step = args.batch_size * args.grad_accum_steps * world_size * args.block_size
                print(f"iter {it}: loss {loss.item() * args.grad_accum_steps:.4f} | lr {lr:.2e} | "
                      f"elapsed {elapsed_min:.1f} min | ~{tokens_per_step:,} tokens/step")

            if is_master and it % args.eval_interval == 0 and it > 0:
                losses = estimate_loss(raw_model, train_data, val_data, args.block_size,
                                        args.batch_size, device, args.eval_iters)
                print(f"  -> eval: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if is_master and it % args.ckpt_interval == 0 and it > 0:
                save_checkpoint(local_ckpt_path, raw_model, optimizer, scaler, config, it, args)
                print(f"  -> saved local checkpoint at iter {it}")
                if args.hf_repo:
                    push_checkpoint(local_ckpt_path, args.hf_repo, args.hf_ckpt_name)
                    print(f"  -> pushed checkpoint to {args.hf_repo}")
    finally:
        # Always try to save on the way out -- finished, hit the time budget,
        # or crashed/Ctrl-C. This is the safety net that keeps a dying
        # session from losing progress since the last ckpt_interval.
        if is_master:
            save_checkpoint(local_ckpt_path, raw_model, optimizer, scaler, config, it, args)
            print(f"Final checkpoint saved to {local_ckpt_path} at iter {it}")
            if args.hf_repo:
                try:
                    push_checkpoint(local_ckpt_path, args.hf_repo, args.hf_ckpt_name)
                    print(f"Final checkpoint pushed to {args.hf_repo}")
                except Exception as e:
                    print(f"Warning: final push to Hugging Face failed ({e}). "
                          f"Local checkpoint is still safe at {local_ckpt_path}.")
        if ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()