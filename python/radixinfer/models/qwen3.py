from ._decoder import build_decoder_model

Qwen3ForCausalLM = build_decoder_model({"has_qk_norm": True})

__all__ = ["Qwen3ForCausalLM"]
