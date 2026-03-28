from __future__ import annotations

from radixinfer.moe.base import BaseMoeBackend

_REGISTRY: dict[str, object] = {
    "fused": lambda: __import__("radixinfer.moe.fused", fromlist=["FusedMoe"]).FusedMoe(),
}


def create_moe_backend(backend: str) -> BaseMoeBackend:
    if backend not in _REGISTRY:
        raise ValueError(
            f"Unknown MoE backend '{backend}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[backend]()  # type: ignore[operator]


__all__ = ["BaseMoeBackend", "create_moe_backend"]
