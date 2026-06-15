from __future__ import annotations

import torch
from tokenizers import Tokenizer

from smol_llm.model import GPT


def generate_text(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("Prompt produced no tokens")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())
