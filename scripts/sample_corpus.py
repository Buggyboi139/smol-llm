#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

SEPARATOR = "<|endoftext|>"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a smaller text sample from monthly corpus files.")
    parser.add_argument("--input-dir", default="data/processed/finenews_text")
    parser.add_argument("--output", default="data/processed/sample_100mb.txt")
    parser.add_argument("--target-mb", type=int, default=100)
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output)
    target_bytes = args.target_mb * 1024 * 1024

    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output}. Use --overwrite to replace it.")

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No input files found in {input_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with output.open("w", encoding="utf-8") as out:
        for file_path in files:
            with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    encoded = line.encode("utf-8")
                    if written + len(encoded) > target_bytes:
                        print(f"Wrote {written / (1024 ** 2):.2f} MiB to {output}")
                        return 0
                    out.write(line)
                    written += len(encoded)

    print(f"Input exhausted. Wrote {written / (1024 ** 2):.2f} MiB to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
