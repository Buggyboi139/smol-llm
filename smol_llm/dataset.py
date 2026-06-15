from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def infer_dtype(path: Path, requested: str = "auto") -> np.dtype:
    if requested != "auto":
        return np.dtype(requested)

    meta_path = path.parent / "data_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "dtype" in meta:
            return np.dtype(meta["dtype"])

    return np.dtype(np.uint16)


class BinaryTokenDataset:
    def __init__(
        self,
        path: str | Path,
        block_size: int,
        device: torch.device | str = "cpu",
        dtype: str = "auto",
    ) -> None:
        self.path = Path(path)
        self.block_size = int(block_size)
        self.device = torch.device(device)
        self.dtype = infer_dtype(self.path, dtype)

        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.data = np.memmap(self.path, dtype=self.dtype, mode="r")
        if len(self.data) <= self.block_size:
            raise ValueError(f"Dataset too small for block_size={self.block_size}: {self.path}")

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self), (batch_size,))
        x = torch.stack([
            torch.from_numpy(np.array(self.data[i : i + self.block_size], dtype=np.int64, copy=True))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(np.array(self.data[i + 1 : i + 1 + self.block_size], dtype=np.int64, copy=True))
            for i in ix
        ])
        return x.to(self.device), y.to(self.device)
