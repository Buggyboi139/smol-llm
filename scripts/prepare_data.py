#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
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


def pick_dtype(vocab_size: int, requested: str) -> np.dtype:
    if requested == "auto":
        return np.dtype(np.uint16 if vocab_size <= 65535 else np.uint32)
    if requested not in {"uint16", "uint32"}:
        raise ValueError("--dtype must be auto, uint16, or uint32")
    dtype = np.dtype(requested)
    if dtype == np.dtype(np.uint16) and vocab_size > 65535:
        raise ValueError("uint16 cannot hold token ids for vocab_size > 65535")
    return dtype


def flush_buffer(buffer: list[int], handle, dtype: np.dtype) -> int:
    if not buffer:
        return 0
    arr = np.asarray(buffer, dtype=dtype)
    arr.tofile(handle)
    count = len(buffer)
    buffer.clear()
    return count


def write_token_stream(
    tokenizer: Tokenizer,
    files: list[Path],
    output_path: Path,
    dtype: np.dtype,
    eos_id: int,
    flush_tokens: int,
) -> int:
    total_tokens = 0
    buffer: list[int] = []

    with output_path.open("wb") as out:
        for file_path in tqdm(files, desc="Encoding files"):
            with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                doc_lines: list[str] = []

                for line in handle:
                    if line.strip() == EOS:
                        text = "".join(doc_lines).strip()
                        doc_lines.clear()
                        if text:
                            buffer.extend(tokenizer.encode(text).ids)
                            buffer.append(eos_id)
                    else:
                        doc_lines.append(line)

                    if len(buffer) >= flush_tokens:
                        total_tokens += flush_buffer(buffer, out, dtype)

                text = "".join(doc_lines).strip()
                if text:
                    buffer.extend(tokenizer.encode(text).ids)
                    buffer.append(eos_id)

                if len(buffer) >= flush_tokens:
                    total_tokens += flush_buffer(buffer, out, dtype)

        total_tokens += flush_buffer(buffer, out, dtype)

    return total_tokens


def copy_token_range(tokens: np.memmap, start: int, end: int, output_path: Path, chunk_tokens: int) -> None:
    with output_path.open("wb") as out:
        cursor = start
        while cursor < end:
            next_cursor = min(cursor + chunk_tokens, end)
            np.asarray(tokens[cursor:next_cursor]).tofile(out)
            cursor = next_cursor


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize text corpus into train.bin and val.bin.")
    parser.add_argument("--input", required=True, help="Input .txt file or directory.")
    parser.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--dtype", default="auto", choices=["auto", "uint16", "uint32"])
    parser.add_argument("--flush-tokens", type=int, default=1_000_000)
    parser.add_argument("--copy-chunk-tokens", type=int, default=10_000_000)
    parser.add_argument("--keep-all-tokens", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be between 0 and 1")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(EOS)
    if eos_id is None:
        raise SystemExit(f"Tokenizer missing {EOS}")

    vocab_size = tokenizer.get_vocab_size()
    dtype = pick_dtype(vocab_size, args.dtype)

    input_path = Path(args.input)
    files = list(iter_text_files(input_path))
    if not files:
        raise SystemExit(f"No text files found: {input_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tokens_path = out_dir / "all_tokens.bin.tmp"
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    meta_path = out_dir / "data_meta.json"

    if all_tokens_path.exists():
        all_tokens_path.unlink()

    total_tokens = write_token_stream(
        tokenizer=tokenizer,
        files=files,
        output_path=all_tokens_path,
        dtype=dtype,
        eos_id=eos_id,
        flush_tokens=args.flush_tokens,
    )

    if total_tokens < 1024:
        raise SystemExit("Not enough tokens produced. Check input data.")

    split = int(total_tokens * (1.0 - args.val_fraction))
    split = max(1, min(split, total_tokens - 1))

    tokens = np.memmap(all_tokens_path, dtype=dtype, mode="r")
    copy_token_range(tokens, 0, split, train_path, args.copy_chunk_tokens)
    copy_token_range(tokens, split, total_tokens, val_path, args.copy_chunk_tokens)
    del tokens

    if not args.keep_all_tokens:
        all_tokens_path.unlink(missing_ok=True)

    meta = {
        "vocab_size": vocab_size,
        "dtype": dtype.name,
        "total_tokens": total_tokens,
        "train_tokens": split,
        "val_tokens": total_tokens - split,
        "val_fraction": args.val_fraction,
        "eos_token": EOS,
        "eos_id": eos_id,
        "input": str(input_path),
        "tokenizer": str(args.tokenizer),
        "train_bin": str(train_path),
        "val_bin": str(val_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Vocab size: {vocab_size}")
    print(f"Dtype: {dtype.name}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Train tokens: {split:,} -> {train_path}")
    print(f"Val tokens: {total_tokens - split:,} -> {val_path}")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
