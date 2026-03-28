from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from radixinfer.distributed import get_tp_info
from radixinfer.utils import div_even


class BaseKVCachePool(ABC):
    """Interface for paged KV cache storage."""

    @abstractmethod
    def k_cache(self, layer_id: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, layer_id: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


class MHAKVCache(BaseKVCachePool):
    """Multi-head attention paged KV cache.

    Physical layout: (2, L, num_pages, page_size, local_kv_heads, head_dim)
    - dim 0: 0=key, 1=value
    - dim 1: layer index
    - dim 2: page index
    - dim 3: position within page
    - dim 4: local KV head count (TP-sharded)
    - dim 5: head dimension
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        # flattened view shape for scatter writes: (num_pages*page_size, local_kv_heads, head_dim)
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, layer_id: int) -> torch.Tensor:
        """Returns (num_pages, page_size, local_kv_heads, head_dim)."""
        return self._k_buffer[layer_id]

    def v_cache(self, layer_id: int) -> torch.Tensor:
        """Returns (num_pages, page_size, local_kv_heads, head_dim)."""
        return self._v_buffer[layer_id]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        """Scatter k/v into the page pool at given linear slot indices."""
        try:
            from radixinfer.kernel import store_cache

            store_cache(
                k_cache=self._k_buffer[layer_id].view(self._storage_shape),
                v_cache=self._v_buffer[layer_id].view(self._storage_shape),
                indices=out_loc,
                k=k,
                v=v,
            )
        except ImportError:
            # Fallback: Python scatter (slower, but correct for testing)
            k_flat = self._k_buffer[layer_id].view(self._storage_shape)
            v_flat = self._v_buffer[layer_id].view(self._storage_shape)
            k_flat[out_loc] = k.view(-1, *self._storage_shape[1:])
            v_flat[out_loc] = v.view(-1, *self._storage_shape[1:])

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
