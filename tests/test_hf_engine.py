import torch

from radixinfer.engine.base import (
    DecodeInput,
    MaterializedBatchMetadata,
    RequestPagedAttentionState,
    RequestTableState,
)
from radixinfer.engine.attention import HuggingFaceFallbackAttentionBackend, PagedAttentionBackend
from radixinfer.engine.hf import HuggingFaceEngine


def test_hf_engine_debug_model_decodes_batch() -> None:
    engine = HuggingFaceEngine("debug", device="cpu")
    output = engine.decode(DecodeInput(request_ids=[1, 2], token_ids=[[1, 2, 3], [4, 5, 6]]))
    assert len(output.next_token_ids) == 2
    assert all(isinstance(token_id, int) for token_id in output.next_token_ids)
    assert len(output.kv_writes) == 2
    assert output.kv_writes[0].token_count == 3


def test_paged_attention_backend_builds_paged_plan_from_request_table_state() -> None:
    backend = PagedAttentionBackend(
        page_size=4,
    )
    metadata = MaterializedBatchMetadata(
        positions=(2, 3),
        input_table_slots=(5, 5),
        input_positions=(2, 3),
        write_table_slots=(5,),
        write_positions=(4,),
        request_token_counts=(2,),
        request_table_states=(
            RequestTableState(
                table_slot=5,
                token_count=3,
                write_position=4,
                page_ids=(8, 8, 8),
                token_ids=(21, 22, 23),
            ),
        ),
        request_paged_states=(
            RequestPagedAttentionState(
                table_slot=5,
                token_count=3,
                write_position=4,
                page_ids=(8,),
                page_indices=(0,),
                kv_page_indices=(0,),
                last_page_len=3,
            ),
        ),
    )
    prepared = backend.prepare_batch(token_ids=[[23, 24]], kv_caches=[None], metadata=metadata)
    assert prepared[0].paged_plan is not None
    assert prepared[0].paged_plan.table_slot == 5
    assert prepared[0].paged_plan.page_ids == (8,)
    assert prepared[0].paged_plan.token_count == 3
    assert prepared[0].paged_plan.write_position == 4
    assert prepared[0].paged_plan.page_indices == (0,)
    assert prepared[0].paged_plan.kv_page_indices == (0,)
    assert prepared[0].paged_plan.last_page_len == 3


def test_hf_fallback_attention_backend_uses_paged_plan() -> None:
    backend = HuggingFaceFallbackAttentionBackend(
        num_layers=2,
        num_heads=2,
        head_dim=8,
        page_size=4,
        device="cpu",
        dtype=torch.float32,
    )
    metadata = MaterializedBatchMetadata(
        positions=(2,),
        input_table_slots=(5,),
        input_positions=(2,),
        write_table_slots=(5,),
        write_positions=(3,),
        request_token_counts=(1,),
        request_table_states=(
            RequestTableState(
                table_slot=5,
                token_count=3,
                write_position=3,
                page_ids=(8, 8, 8),
                token_ids=(21, 22, 23),
            ),
        ),
        request_paged_states=(
            RequestPagedAttentionState(
                table_slot=5,
                token_count=3,
                write_position=3,
                page_ids=(8,),
                page_indices=(0,),
                kv_page_indices=(0,),
                last_page_len=3,
            ),
        ),
    )
    prepared = backend.prepare_batch(token_ids=[[24]], kv_caches=[None], metadata=metadata)
    assert prepared[0].paged_plan is not None
    assert prepared[0].paged_plan.page_ids == (8,)
    assert prepared[0].paged_plan.kv_page_indices == (0,)
    assert prepared[0].input_ids.device.type == "cpu"
