from radixinfer.config import ServerConfig

from .dummy import DummyEngine
from .hf import HuggingFaceEngine


def build_engine(config: ServerConfig):
    if config.engine_kind == "dummy":
        return DummyEngine()
    return HuggingFaceEngine(config.model, device=config.device)


__all__ = ["DummyEngine", "HuggingFaceEngine", "build_engine"]
