from radixinfer.cache.page_pool import KVCacheView
from radixinfer.engine.base import DecodeInput
from radixinfer.engine.dummy import DummyEngine
from radixinfer.engine.hf import HuggingFaceEngine

import torch


def make_kv_view(token_count: int, fill: float, dim: int = 4) -> KVCacheView:
    keys = torch.full((2, token_count, 2, dim), fill_value=fill, dtype=torch.float32)
    values = torch.full((2, token_count, 2, dim), fill_value=fill + 1.0, dtype=torch.float32)
    return KVCacheView(keys=keys, values=values, token_count=token_count)


def test_dummy_engine_output_changes_when_kv_cache_changes() -> None:
    engine = DummyEngine()
    base = engine.decode(DecodeInput(request_ids=[1], token_ids=[[10, 11, 12]])).next_token_ids[0]
    with_kv = engine.decode(
        DecodeInput(
            request_ids=[1],
            token_ids=[[10, 11, 12]],
            kv_caches=[make_kv_view(token_count=3, fill=5.0)],
        )
    ).next_token_ids[0]
    assert base != with_kv


def test_hf_engine_consumes_kv_cache_bias_for_debug_model() -> None:
    engine = HuggingFaceEngine("debug", device="cpu")
    kv = make_kv_view(token_count=3, fill=2.0, dim=8)
    output = engine.decode(
        DecodeInput(
            request_ids=[1],
            token_ids=[[3]],
            kv_caches=[kv],
        )
    )
    assert len(output.next_token_ids) == 1
    assert output.kv_writes[0].token_count == 1
    assert output.kv_writes[0].keys.shape[0] == 2
