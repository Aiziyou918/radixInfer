from radixinfer.engine.base import DecodeInput
from radixinfer.engine.hf import HuggingFaceEngine


def test_hf_engine_debug_model_decodes_batch() -> None:
    engine = HuggingFaceEngine("debug", device="cpu")
    output = engine.decode(DecodeInput(request_ids=[1, 2], token_ids=[[1, 2, 3], [4, 5, 6]]))
    assert len(output.next_token_ids) == 2
    assert all(isinstance(token_id, int) for token_id in output.next_token_ids)
    assert len(output.kv_writes) == 2
    assert output.kv_writes[0].token_count == 3
