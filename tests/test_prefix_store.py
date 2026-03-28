from radixinfer.cache.page_pool import PageSpan
from radixinfer.cache.prefix_store import PrefixStore


def test_prefix_store_matches_page_aligned_prefix() -> None:
    store = PrefixStore(capacity=4, page_size=2)
    store.insert([1, 2, 3, 4, 5], PageSpan(page_ids=(0, 1), token_count=4))
    hit = store.match([1, 2, 3, 4, 9])
    assert hit.matched_tokens == 4
    assert hit.cached_span is not None
    assert hit.cached_span.page_ids == (0, 1)
