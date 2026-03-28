from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import torch
from tqdm import tqdm

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int) -> torch.Tensor:
    """Extract rank r's shard from a tensor, returning a contiguous copy."""
    if any(sub in key for sub in _SPLIT_DIM_0):
        is_kv = any(sub in key for sub in (".k_proj", ".v_proj"))
        if is_kv and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(sub in key for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif "lm_head" in key or "embed_tokens" in key:
        from radixinfer.utils import div_ceil

        num_emb_pp = div_ceil(value.shape[0], n)
        start = r * num_emb_pp
        end = min(start + num_emb_pp, value.shape[0])
        return value[start:end, :].clone()
    else:
        return value


def _get_merge_info(key: str):
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if suffix in key:
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None
    packed_name = match.group("name").removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def _download_hf_weight(model_path: str) -> str:
    """Download or locate HF weights, returning local folder path."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_path, ignore_patterns=["*.bin", "*.pt"])
    except Exception:
        return model_path


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader: yields (name, tensor) pairs, sharded, merged, on device."""
    import safetensors

    from radixinfer.distributed import get_tp_info
    from radixinfer.models.config import ModelConfig

    try:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config = ModelConfig.from_hf(hf_config)
    except Exception as e:
        raise RuntimeError(f"Failed to load HF config from {model_path}: {e}") from e

    model_folder = _download_hf_weight(model_path)
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_folder}")

    tp_info = get_tp_info()
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}

    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                raw = f.get_tensor(name)
                name = name.removeprefix("language_model.")
                tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
                del raw

                merge_info = _get_merge_info(name)
                if merge_info is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = merge_info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe:
                    expert_info = _get_expert_stack_info(out[0])
                    if expert_info is not None:
                        packed_key, expert_idx = expert_info
                        slots = expert_buf.setdefault(packed_key, {})
                        slots[expert_idx] = out[1]
                        if len(slots) != config.num_experts:
                            continue
                        experts = [slots[idx] for idx in range(config.num_experts)]
                        del expert_buf[packed_key]
                        yield packed_key, torch.stack(experts, dim=0)
                        continue

                yield out[0], out[1]

    if merge_buf:
        raise RuntimeError(f"Incomplete merge groups: {list(merge_buf.keys())}")
    if expert_buf:
        raise RuntimeError(f"Incomplete expert tensors: {list(expert_buf.keys())}")
