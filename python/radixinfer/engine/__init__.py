from radixinfer.config import ServerConfig

from .dummy import DummyEngine
from .hf import HuggingFaceEngine


def build_engine(config: ServerConfig):
    if config.engine_kind == "dummy":
        return DummyEngine()
    return HuggingFaceEngine(
        config.model,
        device=config.device,
        kv_num_layers=config.kv_num_layers,
        kv_num_heads=config.kv_num_heads,
        kv_cache_dim=config.kv_cache_dim,
        page_size=config.page_size,
    )


__all__ = ["DummyEngine", "HuggingFaceEngine", "build_engine"]
