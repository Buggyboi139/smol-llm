import torch

from smol_llm.model import GPT, GPTConfig


def test_model_generate_extends_sequence():
    config = GPTConfig(
        vocab_size=64,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 4))

    out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=10)

    assert out.shape == (1, 9)
