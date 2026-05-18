"""
Token sampling from logits.

Supports temperature scaling, top-k filtering, and top-p (nucleus) sampling.
All operations are batched and run on whatever device the logits are on.

Sampling order:
  1. Apply temperature (scale logits)
  2. Apply top-k (zero out all but top k logits)
  3. Apply top-p (zero out lowest-probability tokens until cumulative mass >= p)
  4. Multinomial sample from remaining distribution

With temperature=1.0, top_k=0, top_p=1.0: plain multinomial sampling.
With temperature→0: greedy decoding (argmax).
"""

import torch
import torch.nn.functional as F


def sample(
    logits: torch.Tensor,   # [batch, vocab_size] or [vocab_size]
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Sample one token per logit row.

    Args:
        logits:      Raw (un-normalised) model output.
        temperature: Scale factor applied before softmax.
                     Values < 1 sharpen the distribution (more greedy).
                     Values > 1 flatten it (more random).
        top_k:       If > 0, restrict sampling to the k highest-prob tokens.
        top_p:       If < 1, restrict sampling to the smallest set whose
                     cumulative probability exceeds p (nucleus sampling).

    Returns:
        token_ids: [batch] or scalar, dtype=long.
    """
    scalar_input = logits.dim() == 1
    if scalar_input:
        logits = logits.unsqueeze(0)  # [1, vocab]

    logits = logits.float()  # work in fp32 for numerical stability
    # fp16 lm_head can produce Inf; replace before softmax so we don't get NaN
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

    # Temperature scaling
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    # Near-zero temperature → greedy
    if temperature <= 1e-6:
        token_ids = logits.argmax(dim=-1)
        return token_ids.squeeze(0) if scalar_input else token_ids

    # Top-k: zero out all but top-k logits
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, top_k, dim=-1).values[:, -1:]  # [batch, 1]
        logits = logits.masked_fill(logits < kth_vals, float("-inf"))

    probs = F.softmax(logits, dim=-1)  # [batch, vocab]
    probs = probs.clamp(min=0.0)       # guard against fp rounding giving tiny negatives

    # Top-p nucleus: remove tokens whose cumulative probability exceeds p
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Remove tokens where cumulative prob already exceeds top_p
        # (keep at least one token)
        remove_mask = cumulative - sorted_probs > top_p
        sorted_probs[remove_mask] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
        # Re-normalise after zeroing
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch]

    return token_ids.squeeze(0) if scalar_input else token_ids


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over last dimension — deterministic, no randomness."""
    return logits.argmax(dim=-1)
