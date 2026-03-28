from radixinfer.engine.config import EngineConfig
from radixinfer.engine.engine import Engine, ForwardOutput
from radixinfer.engine.sample import BatchSamplingArgs, Sampler

__all__ = ["Engine", "EngineConfig", "ForwardOutput", "BatchSamplingArgs", "Sampler"]


def build_engine(config):
    """Compatibility shim: build engine from a config or ServerConfig-like object."""
    if hasattr(config, "engine_type") and getattr(config, "engine_type", "hf") == "debug":
        from radixinfer.engine.dummy import DummyEngine

        return DummyEngine()
    # Default to real engine
    return Engine(config)
