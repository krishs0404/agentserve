# Understanding AgentServe Internals

*Filled in from source code audit — Phase 1.*

---

## KV-Cache Structure

Every transformer layer computes attention: each input token produces a Key vector and a Value vector. During autoregressive generation, the model needs to attend to *all previous tokens* on every step — so rather than recompute K and V for all previous tokens every time, we cache them. In AgentServe the KV cache lives as a Python list of `(k, v)` tuples, one tuple per transformer layer, attached directly to the `Request` object as `req.kv_cache`. Each k and v tensor has shape `[T_so_far, n_kv_heads, head_dim]` where T_so_far grows by 1 every decode step. There is also a `BlockAllocator` in `cache.py` that tracks ownership of fixed-size *blocks* of memory using integer IDs — but in the current code that bookkeeping is decoupled from the actual tensor storage. The physical KV tensor pool (`kv_pool`) is never allocated in practice (`allocate_tensor=False`). This is the project's biggest architectural gap.

---

## Prefill vs Decode

**Prefill**: the full prompt (all N tokens) is fed to the model in one forward pass. Because we process all tokens simultaneously, the Q matrix is N×D and the K/V matrices are also N×D, so attention is O(N²) in memory and compute. The output is N logit vectors, but we only care about the last one (which predicts the first output token). After prefill, we save the computed K and V tensors as the KV cache.

**Decode**: each subsequent token is generated one at a time. We feed a single token, look up its KV in the cache (it attends over all T cached tokens), and sample the next token. Because T=1 for the new token, the matmul is 1×D × D×T_total, which is tiny FLOP-wise but requires reading the full KV cache from memory — making it memory-bandwidth-bound.

These phases are handled separately in `engine.py`: `_prefill()` runs the full prompt forward pass and records `first_token_time`; `_decode_step()` feeds one token at a time, appending to the KV cache each call.

---

## Memory Bandwidth Bound

During decode, the model does `1 × hidden_dim` → `attention over T tokens` → `1 logit`. The KV cache for a 1B model at 16-bit precision, with a context of 2048 tokens, is roughly `2 × 16 layers × 2048 tokens × 8 kv_heads × 64 head_dim × 2 bytes ≈ 64 MB`. Each decode step reads that entire 64 MB to generate one token. The compute (matmul) is trivial in comparison — the GPU is waiting on memory reads, not arithmetic. This is why *batching decode steps* matters: if you batch 32 requests, you do 32× the compute while reading nearly the same KV cache from memory (the KV caches are distinct per request, but the model weights are shared and read once). So GPU utilization climbs with batch size during decode, up to memory bandwidth saturation.

---

## Continuous Batching

Static batching: fill a batch of B requests, run the model until all B finish, then fill the next batch. Problem: if request 1 needs 10 output tokens and request 2 needs 1000, the GPU spins generating tokens for request 2 while request 1 has been done for 990 steps. GPU utilization looks fine but you're wasting slots.

Continuous batching (what AgentServe implements via `Scheduler`): requests enter and leave the decode batch as they complete. Every engine step, `get_decode_batch()` returns all currently active requests; when one finishes, its slot is immediately available for a pending request via `get_prefill_batch()`. This keeps GPU utilization high because batch size tracks the actually-active request count, not a padded static size. The cost is complexity: batch size fluctuates, you can't pre-allocate fixed attention buffers, and padding/masking gets more involved.

---

## Paged Attention

With contiguous allocation, each request gets a chunk of memory proportional to `prompt_len + max_output_len`. The problem: you must reserve the full `max_output_len` at arrival time even though you don't know how many tokens will actually be generated. A 256-token request next to a 10-token request leaves 246 tokens of wasted space after the short one finishes — that gap is too small for most new requests. This is external fragmentation.

Paging fixes this: memory is divided into fixed blocks of (say) 16 tokens each. A request gets blocks on demand, one block at a time. Blocks are always the same size, so any free block fits any request. There is still internal fragmentation (the last block might be half-empty) but no external fragmentation. In vLLM's full implementation, the attention kernel directly indexes into a physical block table — the kernel receives a list of block IDs and knows exactly where in GPU memory to find each 16-token chunk. In AgentServe's current implementation, `BlockAllocator` tracks ownership (which request ID owns which block IDs) but the attention computation in `llama.py` uses per-request tensors, not the block pool — so paging exists at the bookkeeping layer but not at the physical memory layer.

---

## Prefix Caching

Many agent requests start with the same system prompt: `"You are an assistant that calls tools..."` followed by the tool definitions. Computing attention over those tokens is identical for every request. If we save the K and V tensors for that prefix after the first request, subsequent requests can skip re-running attention over those tokens entirely.

AgentServe's `prefix_cache.py` implements this at block granularity. Token IDs are split into blocks of `block_size` (default 16). Each block is hashed using blake2b, chained from the previous block's hash — so the hash of block N depends on all blocks 0..N, which means a cache hit at block N implies all previous blocks matched. On a hit, `find_longest_prefix()` returns the block count matched and the block IDs. The engine uses `req.num_cached_tokens = matched_len` to feed only `token_ids[matched_len:]` to the model, skipping the already-cached portion.

**Known gap**: the actual KV tensors for the cached prefix are not threaded into the model's forward pass. `req.kv_cache` starts as `None`, so when the model processes `token_ids[32:]` with `position_offset=32` and `kv_cache=None`, it computes attention over only the un-cached suffix — the model cannot attend to the first 32 tokens. The block IDs returned by the prefix cache are unused by the forward pass. The prefix cache currently only saves *metric counting* (tokens_saved) but not actual compute.

---

## Agent-Aware Scheduling

A chatbot sends one request at a time and waits for the response. An agent fires many requests in parallel — a code-gen call, 5 classification calls, 3 extraction calls — all at once, as tool calls in a DAG step. The agent can't advance to the next DAG step until every tool call in the current step has completed.

Key insight: if the 5 classifications finish in 20 tokens each but the code-gen takes 300 tokens, and the scheduler runs them in FIFO order (code-gen arrived first), the agent waits for code-gen to complete before getting ANY of the classification results. But if we run classifications first (they finish in ~5 steps), the agent has 5 unblocked tool results in 5 steps, and code-gen runs in the background. Total DAG completion time drops significantly even though individual hard requests wait longer.

This is NOT about raw throughput — it's about *total task completion time* as the optimization target, which is different from tokens/second.

---

## Difficulty Classification

The classifier in `difficulty.py` uses keyword matching on the lowercased prompt string. EASY keywords: "classify", "label", "yes or no", "extract the", "fill in the json", etc. HARD keywords: "write a function", "implement", "refactor", "design a system", etc. Prompts with an estimated token count above 2000 (from word count) are hard regardless of keywords. Everything else is MEDIUM.

We don't use an ML classifier because: (1) it would require a second forward pass before the actual inference forward pass, adding latency to every request; (2) keyword matching is deterministic, debuggable, and sub-millisecond; (3) for agent workloads, the task type is usually explicit in the prompt ("classify", "write", "extract"), so heuristics work well. The weakness is brittleness: "Please don't classify this as spam" would hit the EASY classifier even though it's a medium task.

`estimated_output_tokens` (20 for easy, 100 for medium, 256 for hard) is stored on the request but currently only used for metrics. A future improvement would use this estimate for admission control: don't admit a hard request if it would push memory usage over a threshold.

---

## Priority Scheduling (Policy 1)

`_insert_priority()` walks the `pending` deque from back to front to find the insertion point for a new request. Priority 0 (easy) goes before priority 1 (medium) before priority 2 (hard). Within the same priority, FIFO is preserved: a new medium request inserts *after* all existing mediums. This ensures trivial tool calls clear quickly, unblocking downstream DAG steps. The cost is `O(n)` per insert because a Python deque doesn't support random-access insertion — it converts to a list, inserts, converts back. For 1000+ pending requests this becomes visible in profiling.

---

## Batch Admission Control (Policy 2)

`soft_cap = int(max_batch_size * overflow_factor)` where overflow_factor defaults to 1.25. When the decode batch hits `max_batch_size` (8 by default), normally we'd stop admitting requests. But if the next pending request is EASY (priority 0), we admit it up to `soft_cap` (10). The rationale: an easy request generates ~20 tokens, so it will vacate its slot in ~20 steps. The marginal GPU cost of one extra easy request in the decode batch is tiny — one extra row in the KV attention matmul — and the benefit is it unblocks a DAG node. The risk of setting overflow_factor too high: if many easy requests arrive simultaneously, the batch grows beyond what GPU memory can handle (KV caches for all concurrent requests must fit).

---

## Preemption (Policy 3)

`_maybe_preempt()` triggers when: the front of the pending queue is EASY (priority 0), AND the decode batch is at max capacity, AND at least one HARD request in the decode batch has generated fewer than `preempt_after_tokens` (10) output tokens. If conditions are met, the HARD request with the *fewest* output tokens is moved back to pending. Its output tokens are discarded, its KV cache is cleared, and it's re-queued at the front of its priority bucket.

Why only "young" hard requests? Re-prefilling is expensive — the model must process the full prompt again. If a hard request has already generated 200 tokens and is nearly done, preempting it wastes 200 tokens of work plus the re-prefill cost of a long prompt, which likely exceeds any benefit. With < 10 tokens generated, the re-prefill cost is bounded by the prompt length, which is more predictable. The threshold of 10 is a heuristic — a real system would compare estimated remaining work vs re-prefill cost.

---

## Speculative Decoding (future)

The idea: use a small "draft" model (1B params) to generate K tokens speculatively, then verify all K in one forward pass of the large "target" model. If the target would have sampled the same token at position i, accept it; otherwise reject from position i onwards. For easy requests that output short, predictable JSON or single-word answers, the acceptance rate is high (draft and target agree most of the time), so you effectively get K tokens for the price of ~1.2 target forward passes. This is a strong addition for AgentServe because easy requests are exactly the ones speculative decoding works best on.

---

## Attention Masking for Variable-Length Batches

In prefill, the causal mask ensures each token can only attend to earlier tokens. In `llama.py`'s `LlamaAttention.forward()`, when T > 1 (prefill), a `[T, T_total]` mask is constructed: columns 0..offset (the cached prefix) are all zeros (fully attend), and the remaining T×T block has an upper-triangular `-inf` mask. In decode (T=1), the mask is trivially all-attend (the single query attends to all cached tokens). For a heterogeneous batch with requests at different sequence lengths, you'd need padding or block-sparse attention (flash-attn's `varlen` mode). The current implementation processes each request independently in the engine loop rather than truly batching the model forward pass — `_decode_step` calls `model.forward` once per request, not once per batch. This means no GPU parallelism across requests during decode.
