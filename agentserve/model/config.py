"""
Model configuration for Llama 3.2 variants and a tiny debug config.

TinyConfig is used in all tests — runs on CPU in milliseconds with no GPU needed.
The real configs match HuggingFace's published Llama 3.2 architecture.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden_dim: int
    n_heads: int
    n_kv_heads: int  # grouped-query attention: fewer KV heads than Q heads
    n_layers: int
    vocab_size: int
    max_seq_len: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    intermediate_size: int = 0  # SwiGLU hidden dim; set explicitly or computed below

    rope_scaling: dict | None = None
    tie_word_embeddings: bool = False

    @classmethod
    def from_hf_dict(cls, data: dict):
        """Accept the supported Llama architecture; fail on incompatible variants."""
        if data.get("model_type") != "llama":
            raise ValueError("Only Llama checkpoints are supported")
        if data.get("attention_bias", False) or data.get("mlp_bias", False):
            raise ValueError("Biased projections are unsupported")
        if data.get("hidden_act", "silu") != "silu":
            raise ValueError("Only SiLU feedforward activation is supported")
        if (
            data.get("head_dim", data["hidden_size"] // data["num_attention_heads"])
            != data["hidden_size"] // data["num_attention_heads"]
        ):
            raise ValueError("Custom head dimensions are unsupported")
        return cls(
            hidden_dim=data["hidden_size"],
            n_heads=data["num_attention_heads"],
            n_kv_heads=data.get("num_key_value_heads", data["num_attention_heads"]),
            n_layers=data["num_hidden_layers"],
            vocab_size=data["vocab_size"],
            max_seq_len=data["max_position_embeddings"],
            intermediate_size=data["intermediate_size"],
            rms_norm_eps=data.get("rms_norm_eps", 1e-6),
            rope_theta=data.get("rope_theta", 10000.0),
            rope_scaling=data.get("rope_scaling"),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
        )

    def __post_init__(self):
        if self.rope_scaling is not None:
            r = self.rope_scaling
            if r.get("rope_type", r.get("type")) != "llama3":
                raise ValueError("Only default and llama3 RoPE are supported")
            if not (
                r["factor"] >= 1
                and 0 < r["low_freq_factor"] < r["high_freq_factor"]
                and r["original_max_position_embeddings"] > 0
            ):
                raise ValueError("Invalid llama3 RoPE parameters")
        if self.intermediate_size == 0:
            # Llama uses 8/3 * hidden_dim, rounded up to multiple of 256
            raw = int(8 / 3 * self.hidden_dim)
            self.intermediate_size = (raw + 255) // 256 * 256

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.n_heads


# Tiny config for CPU unit tests — forward pass runs in <10ms.
TinyConfig = ModelConfig(
    hidden_dim=64,
    n_heads=4,
    n_kv_heads=2,
    n_layers=2,
    vocab_size=256,
    max_seq_len=512,
    intermediate_size=128,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
)

# Llama 3.1/3.2 frequency-dependent scaling (not linear position scaling).
_LLAMA32_ROPE = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}

# Matches meta-llama/Llama-3.2-1B-Instruct on HuggingFace.
Llama32_1B = ModelConfig(
    hidden_dim=2048,
    n_heads=32,
    n_kv_heads=8,
    n_layers=16,
    vocab_size=128256,
    max_seq_len=131072,
    rope_theta=500000.0,
    intermediate_size=8192,
    rope_scaling=dict(_LLAMA32_ROPE),
    tie_word_embeddings=True,
)

# Matches meta-llama/Llama-3.2-3B-Instruct on HuggingFace.
Llama32_3B = ModelConfig(
    hidden_dim=3072,
    n_heads=24,
    n_kv_heads=8,
    n_layers=28,
    vocab_size=128256,
    max_seq_len=131072,
    rope_theta=500000.0,
    intermediate_size=8192,
    rope_scaling=dict(_LLAMA32_ROPE),
    tie_word_embeddings=True,
)

# Matches meta-llama/Meta-Llama-3-8B-Instruct on HuggingFace.
Llama32_8B = ModelConfig(
    hidden_dim=4096,
    n_heads=32,
    n_kv_heads=8,
    n_layers=32,
    vocab_size=128256,
    max_seq_len=8192,
    rope_theta=500000.0,
    intermediate_size=14336,
)
