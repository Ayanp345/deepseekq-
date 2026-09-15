"""Training loop for the DeepSeek-style model.

Usage
-----
    # train a tokenizer, then the model, on a plain-text corpus
    python training.py --data corpus.txt --train-tokenizer --vocab-size 8000 \
        --preset small --steps 2000 --batch-size 8

    # quick CPU smoke test
    python training.py --data corpus.txt --preset tiny --steps 20 --device cpu
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from model import DeepSeekConfig, DeepSeekForCausalLM, _safe_torch_load
from tokenizer import ByteBPETokenizer

PRESETS: Dict[str, Dict] = {
    # ~1M params - fits on a laptop CPU, useful for smoke tests
    "tiny": dict(hidden_size=128, num_layers=4, num_heads=4, num_kv_heads=2,
                 intermediate_size=384, context_length=256),
    # ~30M params
    "small": dict(hidden_size=512, num_layers=8, num_heads=8, num_kv_heads=2,
                  intermediate_size=1408, context_length=512),
    # ~100M params (default config)
    "base": dict(),
    # MoE variant: ~170M total, ~60M active per token
    "moe": dict(use_moe=True, num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
                moe_intermediate_size=512),
}


def pick_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------- dataset
class PackedTextDataset:
    """Encodes a corpus once and packs it into one contiguous token stream.

    Packing (rather than one-sample-per-chunk) means no padding, no wasted
    compute, and every training example is a full-length window.
    """

    def __init__(self, token_ids: np.ndarray, context_length: int, val_fraction: float = 0.01):
        if len(token_ids) < context_length + 2:
            raise ValueError(
                f"corpus too small: {len(token_ids)} tokens for context_length {context_length}"
            )
        split = max(context_length + 1, int(len(token_ids) * (1 - val_fraction)))
        split = min(split, len(token_ids) - 1)
        self.train = token_ids[:split]
        self.val = token_ids[split:] if len(token_ids) - split > context_length + 1 else None
        self.context_length = context_length

    @classmethod
    def from_text(cls, text: str, tokenizer: ByteBPETokenizer, context_length: int,
                  val_fraction: float = 0.01, add_eos: bool = True) -> "PackedTextDataset":
        ids = tokenizer.encode(text, add_eos=add_eos)
        dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.int32
        return cls(np.asarray(ids, dtype=dtype), context_length, val_fraction)

    def get_batch(self, split: str, batch_size: int, device: torch.device,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
        data = self.train if split == "train" else self.val
        if data is None:
            raise ValueError("no validation split available")
        high = len(data) - self.context_length - 1
        ix = torch.randint(high, (batch_size,), generator=generator)
        x = torch.stack([
            torch.from_numpy(data[i:i + self.context_length].astype(np.int64)) for i in ix
        ])
        return x.to(device, non_blocking=True)


def build_dataset(text_path: str) -> str:
    """Read a corpus file into a single string."""
    with open(text_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# ------------------------------------------------------------------ schedule
def lr_at(step: int, base_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    if step >= total:
        return base_lr * min_ratio
    progress = (step - warmup) / max(total - warmup, 1)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1 - min_ratio) * coeff)


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, device, eval_iters, autocast_ctx) -> Dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        if split == "val" and dataset.val is None:
            continue
        losses = []
        for _ in range(eval_iters):
            batch = dataset.get_batch(split, batch_size, device)
            with autocast_ctx():
                losses.append(model(batch, labels=batch)["loss"].item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


# ---------------------------------------------------------------------- loop
def train(
    model: DeepSeekForCausalLM,
    dataset: PackedTextDataset,
    device: torch.device,
    steps: int = 1000,
    batch_size: int = 8,
    grad_accum: int = 1,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    warmup: int = 100,
    grad_clip: float = 1.0,
    eval_every: int = 100,
    eval_iters: int = 20,
    out_dir: str = "checkpoints",
    log_every: int = 10,
    resume: Optional[str] = None,
    seed: int = 1337,
) -> DeepSeekForCausalLM:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.to(device)
    model.train()
    optimizer = model.configure_optimizers(lr=lr, weight_decay=weight_decay)

    use_cuda = device.type == "cuda"
    amp_dtype = None
    if use_cuda:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    fp16 = amp_dtype == torch.float16
    try:  # torch >= 2.4
        scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    def autocast_ctx():
        if amp_dtype is None:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    start_step, best_val = 0, float("inf")
    if resume and os.path.exists(resume):
        ckpt = _safe_torch_load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val", float("inf"))
        print(f"resumed from {resume} at step {start_step}")

    os.makedirs(out_dir, exist_ok=True)
    tokens_per_step = batch_size * grad_accum * dataset.context_length
    print(f"training {model.num_parameters():,} params | {tokens_per_step:,} tokens/step "
          f"| device={device} | amp={amp_dtype}")

    t0 = time.time()
    for step in range(start_step, steps):
        cur_lr = lr_at(step, lr, warmup, steps)
        for group in optimizer.param_groups:
            group["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(grad_accum):
            batch = dataset.get_batch("train", batch_size, device)
            with autocast_ctx():
                loss = model(batch, labels=batch)["loss"] / grad_accum
            scaler.scale(loss).backward()
            total_loss += loss.item()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step:>6} | loss {total_loss:.4f} | ppl {math.exp(min(total_loss, 20)):.1f} "
                  f"| lr {cur_lr:.2e} | {elapsed:.1f}s")

        if eval_every and step > 0 and step % eval_every == 0:
            metrics = estimate_loss(model, dataset, batch_size, device, eval_iters, autocast_ctx)
            msg = " | ".join(f"{k} {v:.4f}" for k, v in metrics.items())
            print(f"  eval @ {step}: {msg}")
            val = metrics.get("val", metrics.get("train", float("inf")))
            save_checkpoint(model, optimizer, step, val, os.path.join(out_dir, "last.pt"))
            if val < best_val:
                best_val = val
                save_checkpoint(model, optimizer, step, val, os.path.join(out_dir, "best.pt"))
                model.save_pretrained(out_dir)
                print(f"  new best ({val:.4f}) -> {out_dir}")

    save_checkpoint(model, optimizer, steps, best_val, os.path.join(out_dir, "last.pt"))
    model.save_pretrained(out_dir)
    print(f"done in {time.time() - t0:.1f}s -> {out_dir}")
    return model


def save_checkpoint(model, optimizer, step: int, val_loss: float, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "best_val": val_loss,
        "config": model.config.__dict__,
    }, path)


def load_checkpoint(model: DeepSeekForCausalLM, path: str) -> DeepSeekForCausalLM:
    if os.path.exists(path):
        ckpt = _safe_torch_load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model


# ----------------------------------------------------------------------- CLI
def main() -> None:
    p = argparse.ArgumentParser(description="Train the DeepSeek-style model")
    p.add_argument("--data", required=True, help="path to a UTF-8 text corpus")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--preset", default="base", choices=sorted(PRESETS))
    p.add_argument("--train-tokenizer", action="store_true")
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--tokenizer-path", default="tokenizer.json")
    p.add_argument("--context-length", type=int, default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    text = build_dataset(args.data)
    print(f"corpus: {len(text):,} characters")

    if args.train_tokenizer:
        tokenizer = ByteBPETokenizer()
        print(f"training BPE tokenizer to vocab_size={args.vocab_size} ...")
        tokenizer.train(text, args.vocab_size, verbose=True)
        tokenizer.save(args.tokenizer_path)
    elif os.path.exists(args.tokenizer_path):
        tokenizer = ByteBPETokenizer.load(args.tokenizer_path)
    else:
        tokenizer = ByteBPETokenizer()
        tokenizer.save(args.tokenizer_path)
    print(f"tokenizer: {tokenizer}")

    cfg_kwargs = dict(PRESETS[args.preset])
    cfg_kwargs["vocab_size"] = tokenizer.vocab_size     # keep model and tokenizer in sync
    if args.context_length:
        cfg_kwargs["context_length"] = args.context_length
    config = DeepSeekConfig(**cfg_kwargs)

    dataset = PackedTextDataset.from_text(text, tokenizer, config.context_length)
    print(f"tokens: {len(dataset.train):,} train / "
          f"{0 if dataset.val is None else len(dataset.val):,} val")

    model = DeepSeekForCausalLM(config)
    model.gradient_checkpointing = args.grad_checkpointing
    device = pick_device(args.device)

    train(model, dataset, device, steps=args.steps, batch_size=args.batch_size,
          grad_accum=args.grad_accum, lr=args.lr, warmup=args.warmup,
          eval_every=args.eval_every, out_dir=args.out_dir, resume=args.resume,
          seed=args.seed)

    tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))
    print("\nsample:", model.generate_text(tokenizer, "The ", max_new_tokens=64))


if __name__ == "__main__":
    main()
