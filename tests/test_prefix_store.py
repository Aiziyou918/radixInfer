from radixinfer.cache.page_pool import PageSpan
from radixinfer.cache.prefix_store import PrefixStore


def test_prefix_store_matches_page_aligned_prefix() -> None:
    store = PrefixStore(capacity=4, page_size=2)
    key, evicted = store.insert([1, 2, 3, 4, 5], PageSpan(page_ids=(0, 1), token_count=4))
    assert key is not None
    assert evicted == []
    hit = store.match([1, 2, 3, 4, 9])
    assert hit.matched_tokens == 4
    assert hit.cached_span is not None
    assert hit.cached_span.page_ids == (0, 1)
    assert hit.cache_key == key


def test_prefix_store_tracks_lock_refcount() -> None:
    store = PrefixStore(capacity=4, page_size=2)
    key, _ = store.insert([1, 2, 3, 4], PageSpan(page_ids=(0, 1), token_count=4))
    assert key is not None
    assert store.size_info.evictable_size == 4
    assert store.size_info.protected_size == 0
    store.lock(key)
    assert store.entry_ref_count(key) == 1
    assert store.size_info.evictable_size == 0
    assert store.size_info.protected_size == 4
    store.unlock(key)
    assert store.entry_ref_count(key) == 0
    assert store.size_info.evictable_size == 4
    assert store.size_info.protected_size == 0
    store.check_integrity()


def test_prefix_store_evicts_only_unlocked_entries_when_over_capacity() -> None:
    store = PrefixStore(capacity=1, page_size=2)
    key1, evicted1 = store.insert([1, 2], PageSpan(page_ids=(0,), token_count=2))
    assert key1 is not None
    assert evicted1 == []
    store.lock(key1)
    key2, evicted2 = store.insert([3, 4], PageSpan(page_ids=(1,), token_count=2))
    assert key2 is not None
    assert evicted2 == [PageSpan(page_ids=(1,), token_count=2)]
    store.unlock(key1)
    key3, evicted3 = store.insert([5, 6], PageSpan(page_ids=(2,), token_count=2))
    assert key3 is not None
    assert evicted3 == [PageSpan(page_ids=(0,), token_count=2)]
    assert store.size_info.evictable_size == 2
    assert store.size_info.protected_size == 0
    store.check_integrity()


def test_prefix_store_splits_existing_branch_on_partial_match() -> None:
    store = PrefixStore(capacity=4, page_size=2)
    key1, _ = store.insert([1, 2, 3, 4, 5, 6], PageSpan(page_ids=(0, 1, 2), token_count=6))
    assert key1 is not None
    assert store.size_info.evictable_size == 6
    hit = store.match([1, 2, 3, 4, 9, 10])
    assert hit.matched_tokens == 4
    assert hit.cached_span == PageSpan(page_ids=(0, 1), token_count=4)
    assert hit.cache_key is not None
    assert store.size_info.evictable_size == 6
    store.check_integrity()


def test_prefix_store_locking_split_node_protects_ancestors_only_once() -> None:
    store = PrefixStore(capacity=4, page_size=2)
    key, _ = store.insert([1, 2, 3, 4, 5, 6], PageSpan(page_ids=(0, 1, 2), token_count=6))
    assert key is not None
    hit = store.match([1, 2, 3, 4, 9, 10])
    assert hit.cache_key is not None
    store.lock(hit.cache_key)
    assert store.size_info.protected_size == 4
    assert store.size_info.evictable_size == 2
    store.unlock(hit.cache_key)
    assert store.size_info.protected_size == 0
    assert store.size_info.evictable_size == 6
    store.check_integrity()
