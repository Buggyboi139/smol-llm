from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class BinaryTokenDataset:
    def __init__(self, path: str | Path, block_size: int, device: torch.device | str = "cpu") -> None:
        self.path = Path(path)
        self.block_size = int(block_size)
        self.device = torch.device(device)

        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.data) <= self.block_size:
            raise ValueError(f"Dataset too small for block_size={self.block_size}: {self.path}")

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self), (batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64))
            for i in ix
        ])
        return x.to(self.device), y.to(self.device)
