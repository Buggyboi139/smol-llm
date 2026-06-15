import torch

from smol_llm.model import GPT, GPTConfig


def test_model_forward_returns_finite_loss():
    config = GPTConfig(
        vocab_size=128,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0,
    )
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (4, config.block_size))
    y = torch.randint(0, config.vocab_size, (4, config.block_size))

    logits, loss = model(x, y)

    assert logits.shape == (4, config.block_size, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)
