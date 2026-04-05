from ._decoder import build_decoder_model

LlamaForCausalLM = build_decoder_model()

__all__ = ["LlamaForCausalLM"]
