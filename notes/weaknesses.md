# AgentServe — Design Audit: Original Weaknesses and Resolutions

*Phase 1 audit captured during initial build. All three core weaknesses have since been resolved.*

---

## Weakness 1 (RESOLVED): Prefix Cache KV Tensor Reuse

### Original limitation
Early implementation: `PrefixCache.find_longest_prefix()` returned block IDs but not actual KV tensors. The engine called `model.forward()` with `kv_cache=None`, so even on a prefix cache hit the model re-computed attention over all tokens — no compute was actually saved.

### Resolution
`PrefixEntry` now stores `kv_tensors: Optional[list]` — actual `(K, V)` tensor tuples per layer. `find_longest_prefix()` returns a 3-tuple `(matched_len, block_ids, kv_tensors)`. On a hit, the engine seeds `req.kv_cache = cached_kv` before calling `model.forward()`, so the prefill skips those tokens entirely.

After prefill, the engine stores the KV tensors back into the cache:
```python
kv_for_cache = [
    (req.kv_cache[i][0][:complete_blocks_len].detach().clone(),
     req.kv_cache[i][1][:complete_blocks_len].detach().clone())
    for i in range(len(req.kv_cache))
]
self.prefix_cache.store(req.token_ids[:complete_blocks_len], block_ids, kv_for_cache)
```

**Validation**: replaying 50 real SWE-bench sessions (all sharing the same ~14K-token system prompt) achieves a **90% prefix cache hit rate**, confirming real KV reuse is happening.

---

## Weakness 2 (RESOLVED): Decoder Serial Loop

### Original limitation
`engine.step()` called `_decode_step(req)` inside a Python for-loop — one model forward pass per request. For 8 decode-phase requests, that was 8 sequential GPU kernel launches reading model weights 8 times.

### Resolution
The engine now calls `model.forward_decode_batch()` — a single batched forward pass processing all B decode-phase requests simultaneously. KV caches are padded to the max sequence length in the batch and processed in one CUDA kernel launch per layer.

```python
logits_batch, new_kv_caches = self.model.forward_decode_batch(
    last_token_ids=last_tokens,
    kv_caches=kv_caches,
    position_offsets=positions,
)
```

The batched decode is the mechanism that makes the relative batching scheduler (mode e) meaningful: grouping requests with similar predicted output lengths reduces `max(seq_lens)` in each batch step, directly cutting KV-padding compute waste.

---

## Weakness 3 (RESOLVED): Priority Queue O(n) Insertion

### Original limitation
An early version of the scheduler used `list(deque) → bisect insert → deque(list)` — O(n) per arrival. At 1,000+ pending requests this would make queue management the bottleneck.

### Resolution
The scheduler uses three separate O(1) FIFO deques (`_pending_easy`, `_pending_medium`, `_pending_hard`). `add()` is a single `deque.append()` to the right bucket. `_next_pending_candidate()` checks easy → medium → hard in order. No list conversions anywhere.

For relative batching (mode e) and combined batching (mode f), a flat list `_pending_flat` is used with an explicit sliding-window selection — the O(n²) window scan is bounded by `max_prefill_per_step × pending_count` and is acceptable at typical queue depths.

---

## Bonus: Server API Pattern (open)

`server/app.py` still uses a per-request engine loop: each HTTP handler drives its own engine step loop until its request completes. Two consequences: the engine is not thread-safe under concurrent HTTP requests, and requests from different HTTP calls can't be co-batched.

The fix is a background engine tick + async completion events (asyncio.Event per request). Not implemented — the server is used for demonstration; the benchmarks call the engine directly.
