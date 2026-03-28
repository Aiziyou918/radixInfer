from radixinfer.transport.tokenizer_worker import SimpleTokenizer, create_tokenizer_backend


def test_tokenizer_backend_falls_back_to_simple_for_debug() -> None:
    tokenizer = create_tokenizer_backend("debug")
    assert isinstance(tokenizer, SimpleTokenizer)
