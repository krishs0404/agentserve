"""Numerical regression tests: shape checks alone cannot validate KV caching."""
import pytest
import torch

from agentserve.engine.cache import BlockAllocator
from agentserve.model.config import TinyConfig
from agentserve.model.llama import LlamaModel


@pytest.mark.parametrize("lengths", [(4, 4), (2, 5, 3)])
@pytest.mark.parametrize("paged", [False, True])
def test_batched_decode_matches_full_sequence(lengths, paged):
    torch.manual_seed(17)
    model = LlamaModel(TinyConfig).eval()
    sequences = [list(range(1, length + 1)) for length in lengths]
    caches = [model.forward(seq)[1] for seq in sequences]
    request_ids = [str(i) for i in range(len(sequences))]
    allocator = BlockAllocator(
        num_blocks=32, block_size=4, num_layers=TinyConfig.n_layers,
        num_kv_heads=TinyConfig.n_kv_heads, head_dim=TinyConfig.head_dim,
        allocate_tensor=True,
    )
    if paged:
        for rid, cache, seq in zip(request_ids, caches, sequences):
            allocator.allocate(rid, allocator.blocks_needed(len(seq) + 4))
            for layer, (k, v) in enumerate(cache):
                allocator.write_kv_prefill(rid, layer, k, v)

    # Multiple steps catch cache corruption that appears only on the next decode.
    for step in range(3):
        next_tokens = [20 + step + b for b in range(len(sequences))]
        positions = [len(seq) for seq in sequences]
        if paged:
            logits, _ = model.forward_decode_paged(
                next_tokens, positions, positions, allocator, request_ids,
            )
        else:
            logits, caches = model.forward_decode_batch(next_tokens, caches, positions)
        for b, token in enumerate(next_tokens):
            sequences[b].append(token)
            reference_logits, reference_cache = model.forward(sequences[b])
            torch.testing.assert_close(logits[b], reference_logits[-1], atol=2e-5, rtol=2e-5)
            actual_cache = (
                [allocator.gather_kv(request_ids[b], len(sequences[b]), layer)
                 for layer in range(TinyConfig.n_layers)] if paged else caches[b]
            )
            for actual_pair, expected_pair in zip(actual_cache, reference_cache):
                for actual, expected in zip(actual_pair, expected_pair):
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_prefill_and_decode_match_huggingface_reference():
    """Independent tiny Llama reference, random weights, no downloads/GPU needed."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from agentserve.model.loader import _hf_to_ours

    cfg = TinyConfig
    torch.manual_seed(23)
    reference = LlamaForCausalLM(LlamaConfig(
        hidden_size=cfg.hidden_dim, intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.n_layers, num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads, vocab_size=cfg.vocab_size,
        max_position_embeddings=cfg.max_seq_len, rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta, tie_word_embeddings=False,
    )).eval()
    model = LlamaModel(cfg).eval()
    state = {_hf_to_ours(name): tensor for name, tensor in reference.state_dict().items()
             if _hf_to_ours(name) is not None}
    model.load_state_dict(state, strict=True)
    sequences = [[1, 5, 8], [2, 3, 7, 9, 11]]
    caches = []
    for seq in sequences:
        actual, cache = model.forward(seq)
        with torch.inference_mode():
            expected = reference(torch.tensor([seq])).logits[0]
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        caches.append(cache)
    actual, _ = model.forward_decode_batch([12, 13], caches, [len(s) for s in sequences])
    for b, seq in enumerate(sequences):
        with torch.inference_mode():
            expected = reference(torch.tensor([seq + [12 + b]])).logits[0, -1]
        torch.testing.assert_close(actual[b], expected, atol=2e-5, rtol=2e-5)
