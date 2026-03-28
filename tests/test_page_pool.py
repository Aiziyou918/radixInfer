from radixinfer.cache.page_pool import PagePool


def test_page_pool_reserve_and_release() -> None:
    pool = PagePool(total_pages=8, page_size=4)
    reservation = pool.reserve_for_tokens(9)
    assert reservation is not None
    assert len(reservation.page_ids) == 3
    assert pool.free_pages == 5
    pool.release(reservation)
    assert pool.free_pages == 8


def test_page_pool_can_write_and_read_tokens() -> None:
    pool = PagePool(total_pages=4, page_size=4)
    reservation = pool.reserve_for_tokens(6)
    assert reservation is not None
    span = pool.write_tokens(reservation, [1, 2, 3, 4, 5, 6])
    assert span.token_count == 6
    assert pool.read_span(span) == [1, 2, 3, 4, 5, 6]


def test_page_pool_exposes_kv_view() -> None:
    pool = PagePool(total_pages=4, page_size=4, kv_cache_dim=8, kv_num_layers=3, kv_num_heads=2)
    reservation = pool.reserve_for_tokens(3)
    assert reservation is not None
    span = pool.write_tokens(reservation, [10, 20, 30])
    kv = pool.read_kv(span)
    assert kv.token_count == 3
    assert tuple(kv.keys.shape) == (3, 3, 2, 8)
    assert tuple(kv.values.shape) == (3, 3, 2, 8)


def test_page_pool_can_reserve_with_shared_prefix_pages() -> None:
    pool = PagePool(total_pages=8, page_size=2)
    original = pool.reserve_for_tokens(4)
    assert original is not None
    span = pool.write_tokens(original, [1, 2, 3, 4])
    pool.share_span(span)
    shared = pool.reserve_for_tokens(8, prefix_span=span)
    assert shared is not None
    assert list(shared.shared_page_ids) == [0, 1]
    assert len(shared.private_page_ids) == 2
    assert pool.free_pages == 4


def test_shared_pages_are_not_freed_by_owner_release_until_evicted() -> None:
    pool = PagePool(total_pages=4, page_size=2)
    reservation = pool.reserve_for_tokens(4)
    assert reservation is not None
    span = pool.write_tokens(reservation, [1, 2, 3, 4])
    pool.share_span(span)
    pool.release(reservation)
    assert pool.free_pages == 2
    pool.evict_shared(span)
    assert pool.free_pages == 4
