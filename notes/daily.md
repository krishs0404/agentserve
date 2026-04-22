# Daily Learning Journal

Write a few sentences each day about what you learned, what confused you, and what you want to explore next.

---

## YYYY-MM-DD — Project Bootstrap

Built the initial AgentServe skeleton from nano-vLLM.  Read through the nano-vLLM source and noticed:
- The scheduler separates prefill and decode into distinct steps (one or the other per engine tick).
- The block manager uses a hash-based prefix cache — chaining block hashes so a prefix of any length can be detected in O(blocks) time.
- Tensor parallelism is wired through `dist.get_world_size()` everywhere — makes the code hard to read.  We stripped this out.
- CUDA graphs are a key optimization for decode (freezing the GPU kernel launch overhead at constant batch size) — we stripped this too for v0.

Key insight that motivated the project: in nano-vLLM (and vLLM), the scheduler is oblivious to request *difficulty*.  An easy `classify` call and a hard `write-a-program` call get identical treatment.  But in an agent DAG, the easy calls are often *blocking* — the agent can't fire the next tool until all easy calls resolve.  Prioritizing them reduces total DAG latency even if it slightly increases hard-request latency.

Next: run the benchmarks and see if the mock-model metrics show the expected priority effect.

---

*(Add new entries below, newest at top)*
