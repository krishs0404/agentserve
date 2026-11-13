from scripts.convert_lmcache_traces import build_trace


def test_conversion_preserves_dependency_gap_and_unclipped_output():
    sessions = {
        "session": [
            {"input": [{"role": "user", "content": "first"}], "output_length": 1000, "pre_gap": 0},
            {"input": [], "output_length": 10, "pre_gap": 3.75},
        ]
    }
    first, second = build_trace(sessions)
    assert first["depends_on"] is None
    assert second["turn_index"] == 1
    assert second["depends_on"] == 0
    assert second["pre_gap_ms"] == 3750
    assert second["arrival_delay_ms"] == 5750
    assert second["timing_kind"] == "approximate_open_loop_2s_generation"
    assert first["output_length"] == 1000
    assert first["max_tokens"] == 256
