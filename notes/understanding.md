# Understanding AgentServe Internals

*Template — fill in your own explanations as you work through the code.*

---

## KV-Cache Structure

<!-- Explain: what is the KV cache, what shape does it have, why does it exist, what is stored per token vs per layer. -->

---

## Prefill vs Decode

<!-- Explain: what happens in the prefill phase (processing the prompt), what happens in decode (autoregressive generation), why they are handled separately, what "chunked prefill" means. -->

---

## Memory Bandwidth Bound

<!-- Explain: why decode is memory-bandwidth-bound rather than compute-bound, what this means for batching, why processing one token at a time is inefficient. -->

---

## Continuous Batching

<!-- Explain: how continuous batching works (requests join/leave mid-batch), how it differs from static batching, why it improves GPU utilisation. -->

---

## Paged Attention

<!-- Explain: why KV memory is divided into fixed-size blocks instead of allocated contiguously per request, how block tables work, what fragmentation means here. -->

---

## Prefix Caching

<!-- Explain: what a "prefix" is (shared system prompt), how we hash token blocks to detect sharing, how prefix cache hits skip prefill computation. -->

---

## Agent-Aware Scheduling

<!-- Explain: what makes an agent workload different from a chatbot workload, why tool call latency compounds across DAG edges, the core insight that drives this project. -->

---

## Difficulty Classification

<!-- Explain: the heuristic rules used, why we don't use an ML classifier here, the three classes (easy/medium/hard), what estimated_output_tokens is used for. -->

---

## Priority Scheduling

<!-- Explain: Policy 1 in detail — how easy requests are moved to the front, FIFO within same priority, why this helps total agent task completion time. -->

---

## Batch Admission Control

<!-- Explain: Policy 2 in detail — what the soft overflow cap is, why easy requests are worth admitting beyond capacity, what the risk is if we set the cap too high. -->

---

## Preemption

<!-- Explain: Policy 3 in detail — what conditions trigger preemption, why only young hard requests are preemptable, what the cost of preemption is (re-prefill). -->

---

## Speculative Decoding (future)

<!-- Explain: the draft-then-verify idea, why it helps for easy requests, what the acceptance rate measures, how it interacts with paged KV cache. -->

---

## Attention Masking for Variable-Length Batches

<!-- Explain: why a batch of variable-length sequences needs a mask, how the causal mask works, what cu_seqlens means in flash-attention, how prefill and decode use different mask shapes. -->
