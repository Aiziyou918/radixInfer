from radixinfer.cache.prefix_store import PrefixStore


def test_prefix_store_matches_page_aligned_prefix() -> None:
    store = PrefixStore(capacity=4, page_size=2)
    store.insert([1, 2, 3, 4, 5])
    hit = store.match([1, 2, 3, 4, 9])
    assert hit.matched_tokens == 4
