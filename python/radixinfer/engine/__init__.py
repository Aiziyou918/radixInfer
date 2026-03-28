from radixinfer.engine.config import EngineConfig
from radixinfer.engine.engine import Engine, ForwardOutput
from radixinfer.engine.sample import BatchSamplingArgs, Sampler

__all__ = ["Engine", "EngineConfig", "ForwardOutput", "BatchSamplingArgs", "Sampler"]


def build_engine(config):
    """Compatibility shim: build engine from a config or ServerConfig-like object."""
    kind = getattr(config, "engine_kind", None) or getattr(config, "engine_type", "hf")
    if kind in ("dummy", "debug"):
        from radixinfer.engine.dummy import DummyEngine

        return DummyEngine()
    if kind == "hf":
        from radixinfer.engine.hf import HuggingFaceEngine

        model = getattr(config, "model", None) or getattr(config, "model_path", "debug")
        device = getattr(config, "device", "auto")
        return HuggingFaceEngine(model_name=model, device=device)
    # Default to real engine
    return Engine(config)
