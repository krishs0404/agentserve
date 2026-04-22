# Experiment Log

Track predictions, measurements, and surprises here.  Good science: write down what you expect *before* you run the experiment.

| Date | Experiment | Prediction | Measurement | Explanation | Surprise? |
|------|------------|------------|-------------|-------------|-----------|
|      |            |            |             |             |           |

---

## Experiment Template

**Date:** YYYY-MM-DD
**Experiment:** What are you testing? (e.g. "Does priority scheduling reduce easy-request latency?")
**Prediction:** What do you expect to happen, and why?
**Command:**
```bash
# paste the exact command you ran
```
**Measurement:** What did you observe? (paste numbers, not descriptions)
**Explanation:** Why did it happen that way?
**Surprise?** Was anything unexpected? What does it tell you?

---

## Results Table (fill as you run experiments)

### E001 — Priority scheduling latency reduction
**Date:**
**Experiment:** Compare mean latency for easy requests in agent-aware vs baseline mode, 20 requests.
**Prediction:** Agent-aware mode will reduce easy-request latency by 30–50% because easy requests skip the queue.
**Command:**
```bash
uv run python scripts/bench_throughput.py --use-mock --num-requests 20 --compare
```
**Measurement:** *(fill in)*
**Explanation:** *(fill in)*
**Surprise?:** *(fill in)*

---

### E002 — Prefix cache hit rate with shared system prompt
**Date:**
**Experiment:** 20 requests all sharing a 500-token system prompt; measure prefix cache hit rate.
**Prediction:** After the first request warms the cache, subsequent requests should hit the prefix at ~100%.
**Command:**
```bash
uv run python scripts/bench_agent_trace.py --trace traces/synthetic_50.jsonl --compare
```
**Measurement:** *(fill in)*
**Explanation:** *(fill in)*
**Surprise?:** *(fill in)*

---

### E003 — Batch overflow policy for easy requests
**Date:**
**Experiment:** With max_batch_size=4 and overflow_factor=1.25, does an easy request get admitted when batch is full?
**Prediction:** Yes — soft cap of 5 allows one extra easy request.
**Command:**
```bash
uv run python -m pytest tests/test_scheduler.py::TestBatchAdmissionOverflow -v
```
**Measurement:** *(fill in)*
**Explanation:** *(fill in)*
**Surprise?:** *(fill in)*
