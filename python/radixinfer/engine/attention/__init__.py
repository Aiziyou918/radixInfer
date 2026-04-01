from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from .base import BaseAttnBackend, BaseAttnMetadata, HybridBackend
from .fa import FAMetadata, FlashAttentionBackend
from .fi import FIMetadata, FlashInferBackend
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from radixinfer.models.config import ModelConfig


_BACKEND_REGISTRY: Dict[str, type] = {
    "fa": FlashAttentionBackend,
    "fi": FlashInferBackend,
}


def validate_attn_backend(backend: str, allow_auto: bool = True) -> str:
    if backend == "auto":
        assert allow_auto, "auto is not allowed here"
        return backend
    for name in (backend.split(",") if "," in backend else [backend]):
        if name not in _BACKEND_REGISTRY:
            raise ValueError(
                f"Unsupported attention backend '{name}'. Supported: {list(_BACKEND_REGISTRY.keys())}"
            )
    return backend


def create_attention_backend(backend: str, config: "ModelConfig") -> BaseAttnBackend:
    validate_attn_backend(backend, allow_auto=False)
    if "," in backend:
        assert backend.count(",") == 1, "Only one comma allowed in hybrid backend spec"
        prefill_name, decode_name = backend.split(",", 1)
        if prefill_name != decode_name:
            prefill_backend = create_attention_backend(prefill_name, config)
            decode_backend = create_attention_backend(decode_name, config)
            return HybridBackend(prefill_backend, decode_backend)
        backend = prefill_name
    return _BACKEND_REGISTRY[backend](config)


__all__ = [
    "BaseAttnMetadata",
    "BaseAttnBackend",
    "HybridBackend",
    "FAMetadata",
    "FlashAttentionBackend",
    "FIMetadata",
    "FlashInferBackend",
    "create_attention_backend",
    "validate_attn_backend",
    "BaseCaptureData",
]
