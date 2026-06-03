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
    n_kv_heads: int      # grouped-query attention: fewer KV heads than Q heads
    n_layers: int
    vocab_size: int
    max_seq_len: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    intermediate_size: int = 0   # SwiGLU hidden dim; set explicitly or computed below

    def __post_init__(self):
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
