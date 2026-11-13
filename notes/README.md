# Data and historical artifacts

The current reproducible study is documented in
[Output-length prediction](../docs/length-prediction.md). Its raw predictions,
split manifests, and metrics live in
[`experiments/length_prediction/`](../experiments/length_prediction/).

| Artifact | Purpose and status |
|---|---|
| `lmcache_lmcache_787_pairs.jsonl` | Current study input: 24,880 recorded turns, 767 unique sessions. The filename overstates the session count. Truncated user/tool text, output counts, and model/session identifiers; not full contexts or causal timing data. |
| `results_*.json` | Historical exploratory benchmark outputs from before correctness fixes; retained for provenance, requiring reruns. |
| `plots/*.png` | Historical charts derived from exploratory runs; not validated current performance evidence. |
| `findings.md` | Historical interpretations and measurements; superseded by the current report. |
| `weaknesses.md` | Historical implementation audit; resolution and performance claims require revalidation. |

The original converter references `sammshen/lmcache-agentic-traces`; the current
[dataset documentation](https://huggingface.co/datasets/DiscoPosse/lmcache-agentic-traces)
is under a different namespace. The extract has no pinned upstream revision.
The current study records a SHA-256 of its local input; that identifies the exact
bytes used without establishing equivalence to an upstream revision. The dataset
card lists CC BY 4.0; see its attribution and provenance details before reuse.

The terminal demo uses a mock model and artificial step delays. Legacy plotting
scripts and historical charts do not establish GPU or session-latency improvements.
