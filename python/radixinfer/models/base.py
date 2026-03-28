from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from radixinfer.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(BaseOP):
    @abstractmethod
    def forward(self) -> torch.Tensor: ...
