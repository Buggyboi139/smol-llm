from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import trange

from smol_llm.dataset import BinaryTokenDataset
from smol_llm.model import build_model
from smol_llm.utils import append_jsonl, get_device, get_dtype, learning_rate, save_checkpoint, set_seed


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size: int, eval_iters: int) -> dict[str, float]:
    model.eval()
    out = {}
    for split, dataset in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dataset.get_batch(batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = float(losses.mean().item())
    model.train()
    return out


def train(config: dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 1337)))

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    out_cfg = config["output"]

    device = get_device(train_cfg.get("device", "auto"))
    dtype = get_dtype(train_cfg.get("dtype", "auto"), device)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")

    train_data = BinaryTokenDataset(data_cfg["train_bin"], model_cfg["block_size"], device)
    val_data = BinaryTokenDataset(data_cfg["val_bin"], model_cfg["block_size"], device)

    model = build_model(model_cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if bool(train_cfg.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        betas=(0.9, 0.95),
    )

    checkpoint_dir = Path(out_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_cfg.get("log_file", "logs/train_log.jsonl")

    max_steps = int(train_cfg["max_steps"])
    batch_size = int(train_cfg["batch_size"])
    eval_interval = int(train_cfg["eval_interval"])
    eval_iters = int(train_cfg["eval_iters"])
    save_interval = int(train_cfg["save_interval"])

    use_autocast = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and dtype == torch.float16))

    pbar = trange(max_steps, desc="training")
    for step in pbar:
        lr = learning_rate(
            step=step,
            max_steps=max_steps,
            warmup_steps=int(train_cfg.get("warmup_steps", 0)),
            lr=float(train_cfg["learning_rate"]),
            min_lr=float(train_cfg.get("min_learning_rate", train_cfg["learning_rate"])),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = train_data.get_batch(batch_size)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        if step % eval_interval == 0 or step == max_steps - 1:
            losses = estimate_loss(model, train_data, val_data, batch_size, eval_iters)
            record = {"step": step, "lr": lr, "loss": loss.item(), **losses}
            append_jsonl(log_file, record)
            print(f"step={step} train={losses['train']:.4f} val={losses['val']:.4f}")

        if step % save_interval == 0 or step == max_steps - 1:
            payload = {
                "model": model.state_dict(),
                "model_config": model_cfg,
                "config": config,
                "step": step,
            }
            save_checkpoint(checkpoint_dir / "latest.pt", payload)
            save_checkpoint(checkpoint_dir / f"step_{step:07d}.pt", payload)
