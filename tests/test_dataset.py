import torch
import numpy as np

from smol_llm.dataset import BinaryTokenDataset


def test_binary_token_dataset_batch_shapes(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(1000, dtype=np.uint16).tofile(path)

    dataset = BinaryTokenDataset(path, block_size=16, device="cpu")
    x, y = dataset.get_batch(batch_size=8)

    assert x.shape == (8, 16)
    assert y.shape == (8, 16)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
