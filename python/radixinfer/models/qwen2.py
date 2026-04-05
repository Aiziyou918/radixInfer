from ._decoder import build_decoder_model

Qwen2ForCausalLM = build_decoder_model({"has_qk_norm": False, "has_attn_bias": True})

__all__ = ["Qwen2ForCausalLM"]
