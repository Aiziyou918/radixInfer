import torch

from radixinfer.cache.prefix_store import RadixPrefixCache


def _make_cache(page_size: int = 2) -> RadixPrefixCache:
    return RadixPrefixCache(device=torch.device("cpu"), page_size=page_size)


def test_radix_prefix_cache_matches_page_aligned_prefix() -> None:
    cache = _make_cache(page_size=2)
    inserted = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    indices = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)

    cached_len, handle = cache.insert_prefix(inserted, indices)
    assert cached_len == 0
    assert handle.cached_len == 4

    hit = cache.match_prefix(torch.tensor([1, 2, 3, 4, 9], dtype=torch.int32))
    assert hit.cuda_handle.cached_len == 4
    assert torch.equal(
        hit.cuda_handle.get_matched_indices(),
        torch.tensor([10, 11, 12, 13], dtype=torch.int32),
    )


def test_radix_prefix_cache_lock_unlock_updates_size_info() -> None:
    cache = _make_cache(page_size=2)
    _, handle = cache.insert_prefix(
        torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        torch.tensor([20, 21, 22, 23], dtype=torch.int32),
    )

    assert cache.size_info.evictable_size == 4
    assert cache.size_info.protected_size == 0

    cache.lock_handle(handle)
    assert cache.size_info.evictable_size == 0
    assert cache.size_info.protected_size == 4

    cache.lock_handle(handle, unlock=True)
    assert cache.size_info.evictable_size == 4
    assert cache.size_info.protected_size == 0
    cache.check_integrity()


def test_radix_prefix_cache_evict_returns_requested_budget() -> None:
    cache = _make_cache(page_size=2)
    cache.insert_prefix(
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([30, 31], dtype=torch.int32),
    )
    cache.insert_prefix(
        torch.tensor([3, 4, 5, 6], dtype=torch.int32),
        torch.tensor([40, 41, 42, 43], dtype=torch.int32),
    )

    evicted = cache.evict(2)
    assert torch.equal(evicted, torch.tensor([30, 31], dtype=torch.int32))
    assert cache.size_info.evictable_size == 4
    assert cache.size_info.protected_size == 0
    cache.check_integrity()


def test_radix_prefix_cache_rejects_invalid_handle_type() -> None:
    cache = _make_cache(page_size=2)
    try:
        cache.lock_handle(object())  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("lock_handle should reject non-cache handles")
