#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

EOS = "<|endoftext|>"


def iter_text_files(path: Path):
    if path.is_dir():
        yield from sorted(path.glob("*.txt"))
    else:
        yield path


def encode_file(tokenizer: Tokenizer, path: Path, eos_id: int) -> list[int]:
    ids: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        buffer: list[str] = []
        for line in handle:
            if line.strip() == EOS:
                text = "".join(buffer).strip()
                buffer.clear()
                if text:
                    ids.extend(tokenizer.encode(text).ids)
                    ids.append(eos_id)
            else:
                buffer.append(line)
        text = "".join(buffer).strip()
        if text:
            ids.extend(tokenizer.encode(text).ids)
            ids.append(eos_id)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize text corpus into train.bin and val.bin.")
    parser.add_argument("--input", required=True, help="Input .txt file or directory.")
    parser.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--val-fraction", type=float, default=0.005)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(EOS)
    if eos_id is None:
        raise SystemExit(f"Tokenizer missing {EOS}")

    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    input_path = Path(args.input)
    files = list(iter_text_files(input_path))
    if not files:
        raise SystemExit(f"No text files found: {input_path}")

    all_ids: list[int] = []
    for file_path in tqdm(files, desc="Encoding files"):
        all_ids.extend(encode_file(tokenizer, file_path, eos_id))

    if len(all_ids) < 1024:
        raise SystemExit("Not enough tokens produced. Check input data.")

    ids = np.array(all_ids, dtype=dtype)
    split = int(len(ids) * (1.0 - args.val_fraction))
    split = max(1, min(split, len(ids) - 1))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    ids[:split].tofile(train_path)
    ids[split:].tofile(val_path)

    print(f"Vocab size: {vocab_size}")
    print(f"Dtype: {dtype}")
    print(f"Total tokens: {len(ids):,}")
    print(f"Train tokens: {split:,} -> {train_path}")
    print(f"Val tokens: {len(ids) - split:,} -> {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
