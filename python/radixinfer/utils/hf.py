from __future__ import annotations

import functools
import json
import os
from typing import Any


def load_tokenizer(model_path: str):
    """Load a HuggingFace tokenizer, with chat_template fallback for Mistral models."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if not getattr(tokenizer, "chat_template", None):
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass
    return tokenizer


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path, trust_remote_code=True)


def cached_load_hf_config(model_path: str):
    """Load HuggingFace config with memoization."""
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    """Download model weights from HuggingFace Hub if not already local."""
    if os.path.isdir(model_path):
        return model_path
    try:
        from huggingface_hub import snapshot_download
        from tqdm.asyncio import tqdm

        class _DisabledTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                kwargs.pop("name", None)
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)

        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=_DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid HF model ID: {e}"
        )
