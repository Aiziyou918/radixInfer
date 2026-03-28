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
