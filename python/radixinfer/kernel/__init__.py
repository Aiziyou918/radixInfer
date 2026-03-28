"""
Kernel integration module.

Provides Python interfaces to compiled C++/CUDA extension kernels when available,
with Python fallbacks for testing/CI environments.

Exported symbols:
    store_cache     - scatter K/V into paged cache
    indexing        - vocab-parallel embedding lookup
    fast_compare_key - radix tree key comparison
    init_pynccl     - initialize PyNCCL communicator (optional)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Scatter k/v into paged cache at given linear slot indices.

    When the C++ kernel is available it dispatches to `_radixinfer_kernel.store_cache`.
    Otherwise falls back to Python scatter.

    k_cache: (num_pages*page_size, local_kv_heads, head_dim)
    v_cache: same shape as k_cache
    indices: (seq_len,) int32 linear slot indices
    k:       (seq_len, local_kv_heads, head_dim)
    v:       same shape as k
    """
    try:
        import _radixinfer_kernel as _K  # type: ignore[import]

        _K.store_cache(k_cache, v_cache, indices, k, v)
    except ImportError:
        k_cache[indices] = k.view(-1, *k_cache.shape[1:])
        v_cache[indices] = v.view(-1, *v_cache.shape[1:])


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    vocab_range: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Vocab-parallel embedding lookup.

    When TP=1 or vocab_range is None, performs a standard F.embedding.
    With vocab_range, masks out-of-range tokens and returns partial embeddings
    (to be all_reduced across TP ranks).
    """
    import torch.nn.functional as F

    try:
        import _radixinfer_kernel as _K  # type: ignore[import]

        return _K.indexing(weights, indices, vocab_range)
    except ImportError:
        if vocab_range is None:
            return F.embedding(indices, weights)
        start, size = vocab_range
        mask = (indices >= start) & (indices < start + size)
        local_idx = (indices - start).clamp(min=0)
        y = F.embedding(local_idx, weights)
        return y * mask.unsqueeze(-1).to(y.dtype)


def fast_compare_key(
    key_a: torch.Tensor,
    key_b: torch.Tensor,
    length: int,
) -> int:
    """Compare two integer key sequences, returning the common prefix length."""
    try:
        import _radixinfer_kernel as _K  # type: ignore[import]

        return _K.fast_compare_key(key_a, key_b, length)
    except ImportError:
        import torch

        eq = (key_a[:length] == key_b[:length]).to(torch.uint8)
        # first mismatch position
        mismatch = (eq == 0).nonzero()
        return int(mismatch[0].item()) if len(mismatch) > 0 else length


def init_pynccl(
    tp_rank: int,
    tp_size: int,
    tp_cpu_group,
    max_size_bytes: int,
):
    """Initialize a PyNCCL communicator for intra-node tensor parallel all_reduce."""
    try:
        import _radixinfer_kernel as _K  # type: ignore[import]

        return _K.init_pynccl(tp_rank, tp_size, tp_cpu_group, max_size_bytes)
    except ImportError:
        raise RuntimeError(
            "PyNCCL kernel is not available. "
            "Compile the extension or use torch.distributed for TP."
        )


__all__ = ["store_cache", "indexing", "fast_compare_key", "init_pynccl"]
