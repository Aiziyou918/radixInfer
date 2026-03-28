from radixinfer.cache.page_pool import PagePool


def test_page_pool_reserve_and_release() -> None:
    pool = PagePool(total_pages=8, page_size=4)
    reservation = pool.reserve_for_tokens(9)
    assert reservation is not None
    assert len(reservation.page_ids) == 3
    assert pool.free_pages == 5
    pool.release(reservation)
    assert pool.free_pages == 8
