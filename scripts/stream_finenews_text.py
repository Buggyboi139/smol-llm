#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

REPO_ID = "ksolovev/FineNews"
DEFAULT_OUTPUT_DIR = Path("data/processed/finenews_text")
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
SEPARATOR = "\n\n<|endoftext|>\n\n"


def iter_months(start_month: str, end_month: str):
    if not MONTH_RE.match(start_month) or not MONTH_RE.match(end_month):
        raise ValueError("Months must use YYYY-MM format")

    year, month = map(int, start_month.split("-"))
    end_year, end_month_num = map(int, end_month.split("-"))

    while (year, month) <= (end_year, end_month_num):
        yield f"{year}-{month:02d}"
        month += 1
        if month == 13:
            year += 1
            month = 1


def clean_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def write_month(month: str, output_dir: Path, min_chars: int, max_docs: int | None, overwrite: bool) -> dict:
    final_path = output_dir / f"{month}.txt"
    temp_path = output_dir / f"{month}.txt.tmp"
    stats_path = output_dir / f"{month}.stats.json"

    if final_path.exists() and final_path.stat().st_size > 0 and not overwrite:
        return {
            "month": month,
            "status": "skipped_existing",
            "docs_written": None,
            "docs_skipped": None,
            "bytes_written": final_path.stat().st_size,
            "output": str(final_path),
        }

    if temp_path.exists():
        temp_path.unlink()

    pattern = f"{month}/en/*.parquet"
    print(f"\nStreaming {pattern}")

    dataset = load_dataset(REPO_ID, data_files=pattern, split="train", streaming=True)

    try:
        dataset = dataset.select_columns(["text"])
    except Exception:
        pass

    docs_written = 0
    docs_skipped = 0

    with temp_path.open("w", encoding="utf-8") as out:
        for row in tqdm(dataset, desc=month):
            text = clean_text(row.get("text"))
            if len(text) < min_chars:
                docs_skipped += 1
                continue
            out.write(text)
            out.write(SEPARATOR)
            docs_written += 1
            if max_docs is not None and docs_written >= max_docs:
                break

    temp_path.replace(final_path)

    stats = {
        "month": month,
        "status": "complete",
        "docs_written": docs_written,
        "docs_skipped": docs_skipped,
        "bytes_written": final_path.stat().st_size,
        "min_chars": min_chars,
        "max_docs": max_docs,
        "output": str(final_path),
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream FineNews English text without storing parquet files.")
    parser.add_argument("--start-month", default="2021-01")
    parser.add_argument("--end-month", default="2025-12")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--max-docs-per-month", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    months = list(iter_months(args.start_month, args.end_month))

    print("FineNews English streaming extractor")
    print(f"Repo: {REPO_ID}")
    print(f"Output: {output_dir}")
    print(f"Months: {args.start_month} through {args.end_month}")

    if args.dry_run:
        for month in months:
            print(month)
        return 0

    all_stats = []
    for month in months:
        try:
            stats = write_month(month, output_dir, args.min_chars, args.max_docs_per_month, args.overwrite)
            all_stats.append(stats)
            print(f"{month}: {stats['status']} bytes={stats['bytes_written']}")
        except KeyboardInterrupt:
            print("\nInterrupted. Completed monthly files are safe.")
            return 130
        except Exception as exc:
            print(f"\nERROR while processing {month}: {exc}", file=sys.stderr)
            return 1

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(all_stats, indent=2) + "\n", encoding="utf-8")
    total_bytes = sum(item["bytes_written"] for item in all_stats if isinstance(item.get("bytes_written"), int))
    print(f"\nDone. Total output size: {total_bytes / (1024 ** 3):.2f} GiB")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
