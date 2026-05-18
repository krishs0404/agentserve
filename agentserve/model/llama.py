"""
Llama model implementation for AgentServe.

Implements Llama 3.2 architecture from scratch:
  - RMSNorm: root-mean-square layer norm (no learned bias, no mean subtraction)
  - RoPE: rotary positional embeddings applied to Q and K
  - GQA: grouped-query attention — n_kv_heads < n_heads, KV heads are repeated
  - SwiGLU: gated feedforward with SiLU activation
  - Full transformer: embedding → N decoder layers → RMSNorm → LM head

KV-cache design:
  The forward pass accepts an optional kv_cache (list of per-layer (K, V) tensors)
  and returns (logits, updated_kv_cache). This lets the engine store KV state
  per-request without any global GPU memory bookkeeping.

  - Prefill mode: kv_cache=None, processes all prompt tokens in one pass.
  - Decode mode: kv_cache=[(k,v), ...], processes one new token, appends to cache.

Tensor shape conventions (annotated inline):
  B = batch size (1 for autoregressive generation)
  T = number of input tokens this step
  H = n_heads (query heads)
  Kh = n_kv_heads (key/value heads, Kh <= H)
  D = head_dim = hidden_dim // n_heads
  V = vocab_size
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from agentserve.model.config import ModelConfig


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square normalization.

    Unlike LayerNorm, no mean subtraction and no bias.  Cheaper and empirically
    works just as well for large transformers.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # [dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        # Upcast to float32 for numerical stability, then cast back.
        x_f32 = x.float()
        rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f32 * rms).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Rotary positional embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute (cos, sin) rotation matrix for all positions up to max_seq_len.

    Returns a tensor of shape [max_seq_len, head_dim] where the first half is cos
    and the second half is sin (concatenated, matching the apply_rope convention).
    """
    # Inverse frequencies: [head_dim // 2]
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    # positions: [max_seq_len]
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    # outer product: [max_seq_len, head_dim // 2]
    freqs = torch.outer(positions, inv_freq)
    # cos and sin: [max_seq_len, head_dim // 2] each → concat → [max_seq_len, head_dim]
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return cos_sin  # [max_seq_len, head_dim]


def apply_rope(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to tensor x.

    x:      [T, n_heads, head_dim]
    cos_sin:[T, head_dim]  (first half = cos, second half = sin for each position)

    The rotation pairs up dimensions (0,1), (2,3), ..., (D-2,D-1).
    """
    T, n_heads, head_dim = x.shape
    # Split x into pairs: [..., head_dim//2] each
    x1 = x[..., : head_dim // 2]   # [T, n_heads, D//2]
    x2 = x[..., head_dim // 2 :]   # [T, n_heads, D//2]

    # Broadcast cos_sin over n_heads dim
    cos = cos_sin[:, : head_dim // 2].unsqueeze(1)  # [T, 1, D//2]
    sin = cos_sin[:, head_dim // 2 :].unsqueeze(1)  # [T, 1, D//2]

    # Rotate: (x1, x2) → (x1*cos - x2*sin, x2*cos + x1*sin)
    x_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)  # [T, n_heads, D]
    return x_rot.to(x.dtype)


# ---------------------------------------------------------------------------
# Grouped-query attention with KV-cache
# ---------------------------------------------------------------------------

class LlamaAttention(nn.Module):
    """GQA attention layer with optional KV-cache.

    When n_kv_heads < n_heads, each KV head is shared across (n_heads // n_kv_heads)
    query heads.  We expand KV before the attention dot-product.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_groups = config.n_heads // config.n_kv_heads  # GQA repeat factor
        self.scale = self.head_dim ** -0.5

        hidden = config.hidden_dim
        # Q projects to all query heads; K/V project to fewer KV heads
        self.q_proj = nn.Linear(hidden, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, hidden, bias=False)

    def forward(
        self,
        x: torch.Tensor,           # [T, hidden_dim]
        cos_sin: torch.Tensor,     # [T, head_dim]
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        T, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(T, self.n_heads, self.head_dim)     # [T, H, D]
        k = self.k_proj(x).view(T, self.n_kv_heads, self.head_dim)  # [T, Kh, D]
        v = self.v_proj(x).view(T, self.n_kv_heads, self.head_dim)  # [T, Kh, D]

        # Apply rotary embeddings to Q and K (not V)
        q = apply_rope(q, cos_sin)
        k = apply_rope(k, cos_sin)

        # Append to KV cache in decode mode
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=0)  # [T_total, Kh, D]
            v = torch.cat([v_cache, v], dim=0)  # [T_total, Kh, D]

        new_kv_cache = (k, v)  # store for next decode step
        T_total = k.shape[0]

        # Expand KV heads to match Q heads (GQA: repeat each KV head n_groups times)
        if self.n_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_groups, -1).reshape(
                T_total, self.n_heads, self.head_dim
            )  # [T_total, H, D]
            v = v.unsqueeze(2).expand(-1, -1, self.n_groups, -1).reshape(
                T_total, self.n_heads, self.head_dim
            )  # [T_total, H, D]

        # Scaled dot-product attention
        # Rearrange to [H, T, D] for batch matmul
        q = q.transpose(0, 1)       # [H, T, D]
        k = k.transpose(0, 1)       # [H, T_total, D]
        v = v.transpose(0, 1)       # [H, T_total, D]

        # scores: [H, T, T_total]
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale

        # Causal mask: each query position can only attend to past positions
        # In prefill: full lower-triangular mask over the prompt
        # In decode: T=1 so the mask is trivially all-attend (single query)
        if T > 1:
            # [T, T_total] causal mask (T_total >= T when there's a cached prefix)
            mask = torch.full((T, T_total), float("-inf"), device=x.device)
            # Allow attending to cached positions and current causal positions
            offset = T_total - T
            mask[:, :offset] = 0.0  # can attend to all cached prefix tokens
            mask = mask + torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
            scores = scores + mask.unsqueeze(0)  # broadcast over heads

        attn = F.softmax(scores, dim=-1)           # [H, T, T_total]
        out = torch.bmm(attn, v)                   # [H, T, D]
        out = out.transpose(0, 1).reshape(T, -1)   # [T, H*D]

        return self.o_proj(out), new_kv_cache       # [T, hidden_dim]


# ---------------------------------------------------------------------------
# SwiGLU feedforward
# ---------------------------------------------------------------------------

class LlamaMLP(nn.Module):
    """SwiGLU feedforward: gate * SiLU(up) → down.

    Uses three weight matrices (gate, up, down) instead of two.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        inter = config.intermediate_size
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj   = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, hidden_dim]
        gate = F.silu(self.gate_proj(x))   # [T, inter]
        up   = self.up_proj(x)             # [T, inter]
        return self.down_proj(gate * up)   # [T, hidden_dim]


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.attn = LlamaAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.ffn = LlamaMLP(config)

    def forward(
        self,
        x: torch.Tensor,           # [T, hidden_dim]
        cos_sin: torch.Tensor,     # [T, head_dim]
        kv_cache: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        # Pre-norm attention with residual
        h, new_kv = self.attn(self.attn_norm(x), cos_sin, kv_cache)
        x = x + h
        # Pre-norm feedforward with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# Full Llama model
# ---------------------------------------------------------------------------

class LlamaModel(nn.Module):
    """Llama 3.2 transformer.

    forward() handles both prefill and decode in a single code path.
    The caller distinguishes the mode by whether kv_cache is None.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Precompute RoPE table once; reused for all calls
        cos_sin = precompute_rope_freqs(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_table", cos_sin, persistent=False)  # [max_seq_len, head_dim]

    @torch.inference_mode()
    def forward_decode_batch(
        self,
        last_token_ids: list[int],
        kv_caches: list,
        position_offsets: list[int],
    ) -> tuple[torch.Tensor, list]:
        """Batched single-token decode for B requests in one forward pass.

        Processes all requests simultaneously instead of looping over them
        individually, amortising weight reads across the batch.

        Args:
            last_token_ids:   [B] most-recently-generated token for each request.
            kv_caches:        B lists, each is n_layers × (k[L_b, Kh, D], v[L_b, Kh, D]).
            position_offsets: Current absolute sequence position for each request.

        Returns:
            logits:        [B, vocab_size]
            new_kv_caches: B lists of n_layers × (k[L_b+1, Kh, D], v[L_b+1, Kh, D])
        """
        B = len(last_token_ids)
        if B == 0:
            return torch.zeros(0, self.config.vocab_size), []

        device = self.rope_table.device

        tokens = torch.tensor(last_token_ids, dtype=torch.long, device=device)
        x = self.embed(tokens)  # [B, hidden_dim]

        positions = torch.tensor(position_offsets, dtype=torch.long, device=device)
        cos_sin_batch = self.rope_table[positions]  # [B, head_dim]

        new_kv_caches: list = [[] for _ in range(B)]
        Kh = self.config.n_kv_heads
        D = self.config.head_dim
        H = self.config.n_heads
        n_groups = H // Kh

        for layer_idx, layer in enumerate(self.layers):
            layer_kvs = [kv_caches[b][layer_idx] for b in range(B)]
            seq_lens = [int(kv[0].shape[0]) for kv in layer_kvs]
            max_len = max(seq_lens)

            # Pad KV caches to [B, max_len, Kh, D]
            k_padded = x.new_zeros(B, max_len, Kh, D)
            v_padded = x.new_zeros(B, max_len, Kh, D)
            for b, (k_b, v_b) in enumerate(layer_kvs):
                L = seq_lens[b]
                k_padded[b, :L] = k_b.to(dtype=x.dtype, device=x.device)
                v_padded[b, :L] = v_b.to(dtype=x.dtype, device=x.device)

            # Pre-norm, project Q/K/V for new token
            h = layer.attn_norm(x)                              # [B, hidden_dim]
            q   = layer.attn.q_proj(h).view(B, H, D)           # [B, H, D]
            k_n = layer.attn.k_proj(h).view(B, Kh, D)          # [B, Kh, D]
            v_n = layer.attn.v_proj(h).view(B, Kh, D)          # [B, Kh, D]

            # Apply RoPE — apply_rope treats leading dim as sequence dim
            q   = apply_rope(q,   cos_sin_batch)  # [B, H, D]
            k_n = apply_rope(k_n, cos_sin_batch)  # [B, Kh, D]

            # Concatenate new K/V to padded history
            k_full = torch.cat([k_padded, k_n.unsqueeze(1)], dim=1)  # [B, max_len+1, Kh, D]
            v_full = torch.cat([v_padded, v_n.unsqueeze(1)], dim=1)  # [B, max_len+1, Kh, D]
            total_len = max_len + 1

            # GQA expansion: repeat KV heads to match Q heads
            if n_groups > 1:
                k_exp = (k_full.unsqueeze(3)
                               .expand(B, total_len, Kh, n_groups, D)
                               .reshape(B, total_len, H, D))
                v_exp = (v_full.unsqueeze(3)
                               .expand(B, total_len, Kh, n_groups, D)
                               .reshape(B, total_len, H, D))
            else:
                k_exp = k_full  # [B, total_len, H, D]
                v_exp = v_full

            # Batched attention: q [B, H, 1, D] × k [B, H, D, total_len]
            q_4d = q.unsqueeze(2)                   # [B, H, 1, D]
            k_4d = k_exp.transpose(1, 2)            # [B, H, total_len, D]
            v_4d = v_exp.transpose(1, 2)            # [B, H, total_len, D]

            scores = torch.matmul(q_4d, k_4d.transpose(2, 3)) * layer.attn.scale  # [B, H, 1, total_len]

            # Mask padded positions for requests with shorter histories
            min_len = min(seq_lens)
            if min_len < max_len:
                mask = scores.new_zeros(B, 1, 1, total_len)
                for b in range(B):
                    if seq_lens[b] < max_len:
                        mask[b, 0, 0, seq_lens[b]:max_len] = float("-inf")
                scores = scores + mask

            attn = F.softmax(scores.float(), dim=-1).to(x.dtype)  # [B, H, 1, total_len]
            out = torch.matmul(attn, v_4d)                         # [B, H, 1, D]
            out = out.squeeze(2).transpose(1, 2).reshape(B, H * D) # [B, H*D]
            attn_out = layer.attn.o_proj(out)                       # [B, hidden_dim]

            # Pre-norm residuals
            x = x + attn_out
            x = x + layer.ffn(layer.ffn_norm(x))

            # Save unpadded KV for each request
            for b in range(B):
                L = seq_lens[b]
                new_kv_caches[b].append((
                    k_full[b, :L + 1].detach(),
                    v_full[b, :L + 1].detach(),
                ))

        x = self.norm(x)           # [B, hidden_dim]
        logits = self.lm_head(x)   # [B, vocab_size]
        return logits, new_kv_caches

    @torch.inference_mode()
    def forward(
        self,
        token_ids: list[int] | torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            token_ids: prompt tokens (prefill) or single new token (decode).
            kv_cache:  None for prefill; list of (k, v) per layer for decode.
            position_offset: starting position index (= len(prompt) for decode).

        Returns:
            logits:        [T, vocab_size]
            new_kv_cache:  list of (k, v) tensors per layer, updated with new tokens.
        """
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)

        T = token_ids.shape[0]  # number of tokens this step

        # Token embeddings: [T, hidden_dim]
        x = self.embed(token_ids)

        # RoPE frequencies for this step's positions
        positions = torch.arange(position_offset, position_offset + T, device=x.device)
        cos_sin = self.rope_table[positions]  # [T, head_dim]

        # Run through transformer layers, threading KV cache
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            x, layer_new_kv = layer(x, cos_sin, layer_kv)
            new_kv_cache.append(layer_new_kv)

        # Final norm + LM head
        x = self.norm(x)              # [T, hidden_dim]
        logits = self.lm_head(x)      # [T, vocab_size]

        return logits, new_kv_cache


# ---------------------------------------------------------------------------
# Mock model for CPU testing (no real attention)
# ---------------------------------------------------------------------------

class MockModel:
    """Deterministic fake model for tests. Returns random logits, simulates KV shapes."""

    def __init__(self, config: ModelConfig, seed: int = 42):
        self.config = config
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def forward_decode_batch(
        self,
        last_token_ids: list[int],
        kv_caches: list,
        position_offsets: list[int],
    ) -> tuple[torch.Tensor, list]:
        """Batched decode step for MockModel — random logits, simulated KV accumulation."""
        B = len(last_token_ids)
        cfg = self.config
        Kh = cfg.n_kv_heads
        D = cfg.head_dim

        logits = torch.randn(B, cfg.vocab_size, generator=self.rng)

        new_slice_k = torch.zeros(1, Kh, D)
        new_slice_v = torch.zeros(1, Kh, D)

        new_kv_caches = []
        for b in range(B):
            req_kv = []
            for i in range(cfg.n_layers):
                if kv_caches[b] is None:
                    req_kv.append((new_slice_k.clone(), new_slice_v.clone()))
                else:
                    k_cache, v_cache = kv_caches[b][i]
                    req_kv.append((
                        torch.cat([k_cache, new_slice_k], dim=0),
                        torch.cat([v_cache, new_slice_v], dim=0),
                    ))
            new_kv_caches.append(req_kv)

        return logits, new_kv_caches

    def forward(
        self,
        token_ids: list[int] | torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)

        T = token_ids.shape[0]
        cfg = self.config
        Kh = cfg.n_kv_heads
        D = cfg.head_dim

        # Random logits
        logits = torch.randn(T, cfg.vocab_size, generator=self.rng)

        # Simulate KV accumulation: new slice is [T, Kh, D]
        new_slice_k = torch.zeros(T, Kh, D)
        new_slice_v = torch.zeros(T, Kh, D)

        new_kv_cache = []
        for i in range(cfg.n_layers):
            if kv_cache is None:
                new_kv_cache.append((new_slice_k.clone(), new_slice_v.clone()))
            else:
                k_cache, v_cache = kv_cache[i]
                new_kv_cache.append((
                    torch.cat([k_cache, new_slice_k], dim=0),
                    torch.cat([v_cache, new_slice_v], dim=0),
                ))

        return logits, new_kv_cache
