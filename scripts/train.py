#!/usr/bin/env python3

from __future__ import annotations

import argparse

from smol_llm.config import load_config
from smol_llm.trainer import train


def main() -> int:
    parser = argparse.ArgumentParser(description="Train smol-llm.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
