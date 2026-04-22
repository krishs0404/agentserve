"""
Load HuggingFace safetensor weights into our LlamaModel.

Llama 3.2 weight names on HuggingFace map to our module names as follows:

  HF name                                   → our name
  model.embed_tokens.weight                 → embed.weight
  model.layers.{i}.input_layernorm.weight   → layers.{i}.attn_norm.weight
  model.layers.{i}.self_attn.q_proj.weight  → layers.{i}.attn.q_proj.weight
  model.layers.{i}.self_attn.k_proj.weight  → layers.{i}.attn.k_proj.weight
  model.layers.{i}.self_attn.v_proj.weight  → layers.{i}.attn.v_proj.weight
  model.layers.{i}.self_attn.o_proj.weight  → layers.{i}.attn.o_proj.weight
  model.layers.{i}.post_attention_layernorm.weight → layers.{i}.ffn_norm.weight
  model.layers.{i}.mlp.gate_proj.weight     → layers.{i}.ffn.gate_proj.weight
  model.layers.{i}.mlp.up_proj.weight       → layers.{i}.ffn.up_proj.weight
  model.layers.{i}.mlp.down_proj.weight     → layers.{i}.ffn.down_proj.weight
  model.norm.weight                         → norm.weight
  lm_head.weight                            → lm_head.weight
"""

import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

from agentserve.model.llama import LlamaModel


# Maps HuggingFace weight name fragments → our parameter name fragments.
_HF_NAME_MAP = {
    "model.embed_tokens.weight": "embed.weight",
    "model.norm.weight": "norm.weight",
}

_LAYER_MAP = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn.q_proj.weight",
    "self_attn.k_proj.weight": "attn.k_proj.weight",
    "self_attn.v_proj.weight": "attn.v_proj.weight",
    "self_attn.o_proj.weight": "attn.o_proj.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn.gate_proj.weight",
    "mlp.up_proj.weight": "ffn.up_proj.weight",
    "mlp.down_proj.weight": "ffn.down_proj.weight",
}


def _hf_to_ours(hf_name: str) -> str | None:
    """Convert a HuggingFace weight name to our module parameter name."""
    if hf_name in _HF_NAME_MAP:
        return _HF_NAME_MAP[hf_name]
    if "lm_head.weight" in hf_name:
        return "lm_head.weight"
    # Layer weights: model.layers.{i}.{suffix}
    if hf_name.startswith("model.layers."):
        parts = hf_name.split(".", 3)  # ["model", "layers", "{i}", "{suffix}"]
        if len(parts) == 4:
            idx = parts[2]
            suffix = parts[3]
            if suffix in _LAYER_MAP:
                return f"layers.{idx}.{_LAYER_MAP[suffix]}"
    return None  # unrecognised — skip silently


def load_weights(model: LlamaModel, model_dir: str) -> None:
    """Load safetensor weights from a HuggingFace model directory into model."""
    safetensor_files = sorted(glob(os.path.join(model_dir, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    param_dict = dict(model.named_parameters())
    loaded = set()

    for path in safetensor_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for hf_name in f.keys():
                our_name = _hf_to_ours(hf_name)
                if our_name is None:
                    continue
                if our_name not in param_dict:
                    continue
                tensor = f.get_tensor(hf_name)
                param_dict[our_name].data.copy_(tensor)
                loaded.add(our_name)

    missing = set(param_dict.keys()) - loaded
    if missing:
        raise RuntimeError(f"Failed to load weights for parameters: {sorted(missing)}")
