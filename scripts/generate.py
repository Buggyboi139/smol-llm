#!/usr/bin/env python3

from __future__ import annotations

import argparse

import torch

from smol_llm.config import load_config
from smol_llm.generation import generate_text
from smol_llm.model import build_model
from smol_llm.tokenizer import load_tokenizer
from smol_llm.utils import get_device, load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text from a smol-llm checkpoint.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["training"].get("device", "auto"))
    tokenizer = load_tokenizer(config["data"]["tokenizer"])
    checkpoint = load_checkpoint(args.checkpoint, device)

    model_cfg = checkpoint.get("model_config", config["model"])
    model = build_model(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])

    with torch.no_grad():
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
