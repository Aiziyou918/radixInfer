from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from radixinfer.cache.page_pool import KVCacheView

from .base import AttentionCacheWrite, MaterializedBatchMetadata


@dataclass(frozen=True)
class AttentionInputs:
    input_ids: torch.Tensor
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None
    metadata: MaterializedBatchMetadata | None = None


class AttentionBackend:
    def prepare_batch(
        self,
        token_ids: list[list[int]],
        kv_caches: list[KVCacheView | None] | None = None,
        metadata: MaterializedBatchMetadata | None = None,
    ) -> list[AttentionInputs]:
        raise NotImplementedError

    def extract_cache_writes(
        self,
        model_outputs: list[Any],
        input_lengths: list[int],
    ) -> list[AttentionCacheWrite]:
        raise NotImplementedError


@dataclass
class HuggingFaceAttentionBackend(AttentionBackend):
    num_layers: int
    num_heads: int
    head_dim: int
    device: str
    dtype: torch.dtype

    def prepare_batch(
        self,
        token_ids: list[list[int]],
        kv_caches: list[KVCacheView | None] | None = None,
        metadata: MaterializedBatchMetadata | None = None,
    ) -> list[AttentionInputs]:
        prepared: list[AttentionInputs] = []
        kv_caches = kv_caches or [None] * len(token_ids)
        for index, (request_tokens, kv_cache) in enumerate(zip(token_ids, kv_caches, strict=True)):
            input_ids = torch.tensor([request_tokens], dtype=torch.long, device=self.device)
            request_metadata = None
            if metadata is not None:
                request_metadata = MaterializedBatchMetadata(
                    positions=[metadata.positions[index]] if index < len(metadata.positions) else [],
                    input_table_slots=(
                        [metadata.input_table_slots[index]] if index < len(metadata.input_table_slots) else []
                    ),
                    input_positions=(
                        [metadata.input_positions[index]] if index < len(metadata.input_positions) else []
                    ),
                    write_table_slots=(
                        [metadata.write_table_slots[index]] if index < len(metadata.write_table_slots) else []
                    ),
                    write_positions=(
                        [metadata.write_positions[index]] if index < len(metadata.write_positions) else []
                    ),
                )
            if kv_cache is None or kv_cache.token_count == 0:
                prepared.append(
                    AttentionInputs(input_ids=input_ids, past_key_values=None, metadata=request_metadata)
                )
                continue
            legacy_cache = self._to_past_key_values(kv_cache)
            prepared.append(
                AttentionInputs(
                    input_ids=input_ids,
                    past_key_values=DynamicCache.from_legacy_cache(legacy_cache),
                    metadata=request_metadata,
                )
            )
        return prepared

    def extract_cache_writes(
        self,
        model_outputs: list[Any],
        input_lengths: list[int],
    ) -> list[AttentionCacheWrite]:
        writes: list[AttentionCacheWrite] = []
        for model_output, input_length in zip(model_outputs, input_lengths, strict=True):
            if not getattr(model_output, "past_key_values", None):
                writes.append(
                    AttentionCacheWrite(keys=torch.empty(0), values=torch.empty(0), token_count=0)
                )
                continue
            layer_keys: list[torch.Tensor] = []
            layer_values: list[torch.Tensor] = []
            for layer_key, layer_value in model_output.past_key_values:
                key_delta = layer_key[:, :, -input_length:, :].permute(0, 2, 1, 3).squeeze(0)
                value_delta = layer_value[:, :, -input_length:, :].permute(0, 2, 1, 3).squeeze(0)
                layer_keys.append(key_delta.to(dtype=torch.float32, device="cpu"))
                layer_values.append(value_delta.to(dtype=torch.float32, device="cpu"))
            writes.append(
                AttentionCacheWrite(
                    keys=torch.stack(layer_keys, dim=0),
                    values=torch.stack(layer_values, dim=0),
                    token_count=input_length,
                )
            )
        return writes

    def _to_past_key_values(
        self,
        kv_cache: KVCacheView,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        layers = min(self.num_layers, kv_cache.keys.shape[0])
        heads = min(self.num_heads, kv_cache.keys.shape[2])
        dim = min(self.head_dim, kv_cache.keys.shape[3])
        past: list[tuple[torch.Tensor, torch.Tensor]] = []
        token_count = kv_cache.token_count
        for layer_idx in range(self.num_layers):
            key = torch.zeros(
                (1, self.num_heads, token_count, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            value = torch.zeros_like(key)
            if layer_idx < layers and token_count > 0:
                src_key = kv_cache.keys[layer_idx, :token_count, :heads, :dim].permute(1, 0, 2)
                src_value = kv_cache.values[layer_idx, :token_count, :heads, :dim].permute(1, 0, 2)
                key[:, :heads, :, :dim] = src_key.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                value[:, :heads, :, :dim] = src_value.to(device=self.device, dtype=self.dtype).unsqueeze(0)
            past.append((key, value))
        return tuple(past)
