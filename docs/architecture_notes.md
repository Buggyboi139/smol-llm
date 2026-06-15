# Architecture notes

The model now uses a small modern decoder-only transformer layout instead of a GPT-2-style baseline.

## Current default choices

- Rotary positional embeddings instead of learned absolute position embeddings.
- RMSNorm instead of LayerNorm.
- SwiGLU feed-forward blocks instead of GELU MLP blocks.
- Grouped-query attention support through `n_kv_head`.
- PyTorch scaled dot product attention instead of hand-written attention math.
- Byte-level BPE tokenizer with `<|endoftext|>` as the document separator.

## Why

These choices are common in modern small and large decoder-only LLMs because they improve training stability, parameter efficiency, and attention performance without making the educational implementation unreadable.

## References

- PyTorch scaled dot product attention: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch AMP: https://docs.pytorch.org/docs/stable/amp.html
- Hugging Face dataset streaming: https://huggingface.co/docs/datasets/stream
- RoFormer / RoPE: https://arxiv.org/abs/2104.09864
- Llama 2: https://arxiv.org/abs/2307.09288
- Mistral 7B: https://arxiv.org/abs/2310.06825
