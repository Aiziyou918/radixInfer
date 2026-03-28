from radixinfer.models.base import BaseLLMModel
from radixinfer.models.config import ModelConfig, RotaryConfig
from radixinfer.models.register import get_model_class
from radixinfer.models.weight import load_weight


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    return get_model_class(model_config.architectures[0], model_config)


__all__ = ["create_model", "load_weight", "BaseLLMModel", "ModelConfig", "RotaryConfig"]
