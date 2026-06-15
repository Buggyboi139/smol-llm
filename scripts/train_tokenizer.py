#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
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
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train(files, trainer)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise SystemExit("Tokenizer training failed to create <|endoftext|>")

    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        special_tokens=[("<|endoftext|>", eos_id)],
    )

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    config_path = output_dir / "tokenizer_config.json"
    tokenizer_config = {
        "tokenizer_file": str(tokenizer_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "requested_vocab_size": args.vocab_size,
        "eos_token": "<|endoftext|>",
        "eos_id": eos_id,
        "type": "byte_level_bpe",
    }
    config_path.write_text(json.dumps(tokenizer_config, indent=2) + "\n", encoding="utf-8")

    print(f"Saved tokenizer: {tokenizer_path}")
    print(f"Saved config: {config_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"EOS id: {eos_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
