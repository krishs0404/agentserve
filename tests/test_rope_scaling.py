"""Independent Hugging Face reference for scaled RoPE and checkpoint loading."""

from dataclasses import replace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from agentserve.model.config import Llama32_1B, ModelConfig
from agentserve.model.llama import LlamaModel, precompute_rope_freqs
from agentserve.model.loader import load_weights


def reference_config():
    return LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=16384,
        rope_theta=500000.0,
        rope_scaling=Llama32_1B.rope_scaling,
        tie_word_embeddings=True,
    )


def test_scaled_rope_matches_hf_across_frequency_bands_and_long_positions():
    cfg = reference_config()
    rotary = LlamaRotaryEmbedding(cfg)
    positions = torch.tensor([[0, 1, 127, 8191, 8192, 12000]])
    cos, sin = rotary(torch.zeros(1, 1, 16), positions)
    table = precompute_rope_freqs(16, 16384, cfg.rope_theta, cfg.rope_scaling)
    torch.testing.assert_close(table[positions[0], :8], cos[0, :, :8])
    torch.testing.assert_close(table[positions[0], 8:], sin[0, :, :8])
    ordinary = precompute_rope_freqs(16, 16384, cfg.rope_theta)
    assert not torch.allclose(table[8192], ordinary[8192])


def test_scaled_checkpoint_loading_and_repeated_decode(tmp_path):
    torch.manual_seed(31)
    reference = LlamaForCausalLM(reference_config()).eval()
    reference.save_pretrained(tmp_path)
    config = ModelConfig.from_hf_dict(reference.config.to_dict())
    model = LlamaModel(config).eval()
    load_weights(model, str(tmp_path))
    # Long absolute positions test scaling without quadratic long-context attention.
    seq = [1, 7, 11]
    offset = 8190
    actual, cache = model.forward(seq, position_offset=offset)
    for _ in range(4):
        with torch.inference_mode():
            expected = reference(
                torch.tensor([seq]), position_ids=torch.arange(offset, offset + len(seq))[None]
            ).logits[0]
        torch.testing.assert_close(actual[-1], expected[-1], atol=2e-5, rtol=2e-5)
        token = int(expected[-1].argmax())
        assert int(actual[-1].argmax()) == token
        actual, cache = model.forward([token], cache, offset + len(seq))
        seq.append(token)
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_weights(LlamaModel(replace(config, rope_scaling=None)), str(tmp_path))


def test_unsupported_checkpoint_configuration_fails():
    data = reference_config().to_dict()
    data["rope_scaling"] = {"rope_type": "linear", "factor": 2}
    with pytest.raises(ValueError, match="RoPE"):
        ModelConfig.from_hf_dict(data)
