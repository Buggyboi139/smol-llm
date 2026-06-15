from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

EOS_TOKEN = "<|endoftext|>"


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def eos_id(tokenizer: Tokenizer) -> int:
    value = tokenizer.token_to_id(EOS_TOKEN)
    if value is None:
        raise ValueError(f"Tokenizer is missing {EOS_TOKEN}")
    return int(value)


def encode(tokenizer: Tokenizer, text: str, add_eos: bool = False) -> list[int]:
    ids = tokenizer.encode(text).ids
    if add_eos:
        ids.append(eos_id(tokenizer))
    return ids


def decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)
