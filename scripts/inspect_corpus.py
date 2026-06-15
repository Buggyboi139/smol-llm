#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a text corpus directory or file.")
    parser.add_argument("path", nargs="?", default="data/processed/finenews_text")
    parser.add_argument("--pattern", default="*.txt")
    args = parser.parse_args()

    path = Path(args.path)
    files = [path] if path.is_file() else sorted(path.glob(args.pattern))

    if not files:
        raise SystemExit(f"No files found at {path}")

    total_bytes = 0
    total_docs = 0

    for file_path in files:
        size = file_path.stat().st_size
        total_bytes += size
        docs = 0
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip() == "<|endoftext|>":
                    docs += 1
        total_docs += docs
        print(f"{file_path}: {human_bytes(size)} docs={docs}")

    print("-")
    print(f"files={len(files)}")
    print(f"total_size={human_bytes(total_bytes)}")
    print(f"total_docs={total_docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
