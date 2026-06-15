#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math

import torch

from smol_llm.config import load_config
from smol_llm.dataset import BinaryTokenDataset
from smol_llm.model import build_model
from smol_llm.trainer import estimate_loss
from smol_llm.utils import get_device, load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate smol-llm validation loss.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["training"].get("device", "auto"))
    checkpoint = load_checkpoint(args.checkpoint, device)

    model_cfg = checkpoint.get("model_config", config["model"])
    model = build_model(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])

    train_data = BinaryTokenDataset(config["data"]["train_bin"], model_cfg["block_size"], device)
    val_data = BinaryTokenDataset(config["data"]["val_bin"], model_cfg["block_size"], device)

    losses = estimate_loss(model, train_data, val_data, config["training"]["batch_size"], args.iters)
    print(f"train_loss={losses['train']:.4f}")
    print(f"val_loss={losses['val']:.4f}")
    print(f"val_ppl={math.exp(losses['val']):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
