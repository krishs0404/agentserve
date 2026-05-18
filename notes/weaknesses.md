# AgentServe — Three Critical Weaknesses

*Phase 1 audit. These are the places where a serious systems engineer would push back hard.*

---

## Weakness 1: The Prefix Cache Is Fake — It Doesn't Actually Reuse KV Tensors

### What the code does
`PrefixCache.find_longest_prefix()` returns `(matched_token_count, block_ids)`. The engine sets `req.num_cached_tokens = matched_len` and then calls:

```python
input_ids = req.token_ids[req.num_cached_tokens:]
logits, kv_cache = self.model.forward(
    token_ids=input_ids,
    kv_cache=req.kv_cache,   # ← this is None at prefill start
    position_offset=req.num_cached_tokens,
)
```

`req.kv_cache` is `None` for all new requests. So even on a cache hit, the model forward pass receives `kv_cache=None`, which means no cached KV tensors are injected. The model processes only `token_ids[32:]` with `position_offset=32` — correct positions, but NO KV context for tokens 0..31. The transformer can't attend to the first 32 tokens because their K/V vectors were never fed in.

### Why this is wrong
Attention requires that every previous token's K and V vectors be present. If you skip tokens 0..31 from the forward pass without providing their cached K/V, the model produces semantically wrong outputs (it can't attend to the system prompt). The `prefix_tokens_saved` metric is being incremented but no actual compute is saved and the outputs are broken for any request with a genuine prefix cache hit.

### What the correct fix looks like
The prefix cache must store the actual (K, V) tensors for cached blocks, not just block IDs. On a hit, you retrieve those tensors, and pass them as the initial `kv_cache` argument to the model forward pass. This is what vLLM's PagedAttention does: the attention kernel receives a block table (list of physical block IDs), and the CUDA kernel directly reads K/V from those physical memory locations. In our Python-level implementation the fix is simpler: `PrefixEntry` should store `kv_tensors: list[tuple[Tensor, Tensor]]` (one per layer), and `_prefill()` should initialize `req.kv_cache = cached_kv_tensors` before the forward call.

**Cost of fix**: memory — storing KV tensors in the prefix cache means they live twice (in the cache and potentially in the request's own kv_cache after prefill extends it). You'd need copy-on-write or a shared reference scheme. But for a prototype, storing copies is fine.

---

## Weakness 2: The Decoder Doesn't Actually Batch — It Loops

### What the code does
`engine.step()` calls `_decode_step(req)` inside a Python for-loop:

```python
for req in decode_batch:
    done = self._decode_step(req)
```

And `_decode_step` calls `self.model.forward(token_ids=[last_token], ...)` — one model forward pass per request, per step. For 8 requests in the decode batch, that's 8 sequential forward passes.

### Why this is wrong
The entire point of batched inference is to execute ONE forward pass for multiple requests simultaneously. The GPU is designed for wide matrix operations — a single forward pass for batch=8 does roughly the same KV cache reads as batch=1 (the model weights are read once, KV caches are distinct) but generates 8 tokens. In the current loop, you read the model weights 8 times sequentially. This gives ~1/8th the throughput of actual batched decode.

In vLLM/SGLang, all decode-phase requests are concatenated into a single batched tensor: `token_ids` is a `[B, 1]` batch, and the attention kernel processes all B requests in one CUDA kernel launch using a ragged KV cache (paged memory, block tables). AgentServe processes them one by one.

### What the correct fix looks like
Stack the decode tokens into a batch: `batch_tokens = torch.tensor([req.output_token_ids[-1] for req in decode_batch])` and call `model.forward(batch_tokens, ...)` once. The model needs to support batched decode, which requires batching the KV cache lookup — either as a padded tensor or (properly) via paged attention with block tables. This is non-trivial but the model architecture already supports `B > 1` implicitly through the attention matmul; you'd need to handle variable-length KV caches across requests.

---

## Weakness 3: Priority Queue Insertion Is O(n) and Breaks Under Load

### What the code does
`_insert_priority()` converts the `deque` to a `list`, finds the insertion point by scanning backwards, inserts, and converts back:

```python
lst = list(self.pending)
lst.insert(insert_at, request)
self.pending = deque(lst)
```

Both `list(deque)` and `deque.insert` are O(n). At steady-state with 1000 pending requests, each new arrival does O(n) work just to join the queue.

### Why this matters
Under realistic load (100+ concurrent requests), the scheduler becomes the bottleneck. The engine runs in a Python loop; if each step involves O(n) list manipulation for every newly arriving request, you can saturate the CPU doing queue management while the GPU sits idle. The mock benchmark won't surface this because it runs serially, but a concurrent load test with 1000 requests will.

Also: the `baseline_mode=True` path uses `deque.append` (O(1)), so benchmarks comparing agent-aware vs baseline will conflate scheduling overhead with policy benefit.

### What the correct fix looks like
Use three separate FIFO queues, one per priority level:

```python
self._queues = {0: deque(), 1: deque(), 2: deque()}  # easy, medium, hard

def add(self, req):
    self._queues[req.priority].append(req)

def _next_pending(self):
    for p in (0, 1, 2):
        if self._queues[p]:
            return self._queues[p][0]
    return None
```

This is O(1) for all operations, preserves FIFO within priority levels, and is simpler to reason about. The current O(n) approach was probably chosen to avoid maintaining three separate state queues, but the simplicity cost isn't worth the O(n) penalty.

---

## Bonus: The Server API Defeats Continuous Batching

Not technically in `engine/` but worth flagging: `server/app.py` calls `engine.step()` in a tight loop, one step per `await asyncio.to_thread(engine.step)`. Two problems:

1. **Single-request loops**: each HTTP handler drives its own engine loop for its own request. A second concurrent HTTP request starts its own loop. Both loops now call `engine.step()` concurrently from different threads — the engine is not thread-safe (shared `_incoming` deque, shared `scheduler`, shared `completed_requests`).

2. **Continuous batching only works if requests arrive together**: the engine's scheduling benefit comes from seeing multiple requests at once and picking the right ones to prefill/decode. With the current server, each request submits to the engine and immediately drives the loop until completion. This serializes requests that could have been batched.

The right pattern for an async server: a background task drives the engine loop at a fixed tick rate (or continuously), and HTTP handlers just submit requests and await a completion event (asyncio.Event or a queue). All requests accumulate in `_incoming` between ticks, and the engine sees them all at once.
