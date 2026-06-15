#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<|endoftext|>"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", required=True, help="Input text file or directory of .txt files.")
    parser.add_argument("--output-dir", default="data/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        files = sorted(str(path) for path in input_path.glob("*.txt"))
    else:
        files = [str(input_path)]

    if not files:
        raise SystemExit(f"No input files found: {input_path}")

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train(files, trainer)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        special_tokens=[("<|endoftext|>", eos_id)],
    )

    out_path = output_dir / "tokenizer.json"
    tokenizer.save(str(out_path))

    print(f"Saved tokenizer: {out_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"EOS id: {eos_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
