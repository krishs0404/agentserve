"""Real-model benchmarks must never silently use character token IDs."""
import pytest
from scripts.bench_ablation import build_tokenizer
from scripts.bench_trajectories import _make_request


def test_mock_tokenizer_is_explicit():
    assert build_tokenizer(None, True)("abc") == [97, 98, 99]


def test_real_mode_requires_model_directory():
    with pytest.raises(ValueError, match="model"):
        build_tokenizer(None, False)


def test_real_tokenizer_load_failure_is_fatal(monkeypatch):
    from transformers import AutoTokenizer

    def fail(*args, **kwargs):
        raise OSError("tokenizer files missing")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fail)
    with pytest.raises(RuntimeError, match="tokenizer"):
        build_tokenizer("local-model", False)


def test_trajectory_request_uses_supplied_tokenizer():
    req = _make_request("hello", 8, "trajectory-1", 0, 3, 100.,
                        tokenize=lambda text: [101, 202, 303])
    assert req.token_ids == [101, 202, 303]
