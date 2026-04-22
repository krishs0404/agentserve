"""
Generate a synthetic agent-like workload.

Produces 50 requests that simulate what an agent running tool calls looks like:
  - All share a 500-token system prompt (tests prefix caching)
  - 60% easy: classify, extract, yes/no
  - 25% medium: summarize, short code
  - 15% hard: long code gen, multi-step planning

Requests arrive in "bursts" of 5-8 at a time with short inter-burst delays,
mimicking an agent dispatching parallel tool calls.

Output: traces/synthetic_50.jsonl with one JSON record per line.
Fields:
  request_id, prompt, arrival_delay_ms, category, expected_difficulty
"""

import json
import random
import sys
import os

# Make agentserve importable from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "traces", "synthetic_50.jsonl")

# A 500-token system prompt shared by all requests.
# (~500 tokens ≈ 375 words at 0.75 words/token)
SYSTEM_PROMPT = (
    "You are a highly capable AI assistant specialized in software engineering tasks. "
    "You have access to a set of tools including code execution, file reading, web search, "
    "and database queries. Your goal is to complete tasks accurately and efficiently.\n\n"
    "When answering questions:\n"
    "- Be concise and precise.\n"
    "- If asked to classify, output only the label.\n"
    "- If asked to extract, output only the extracted value.\n"
    "- If asked to write code, produce clean, well-commented code.\n"
    "- If asked to plan, produce a numbered step-by-step plan.\n\n"
    "Context: You are working inside an automated agent pipeline. Responses may be parsed "
    "programmatically. Do not include conversational filler. Do not apologize. Do not say "
    "'certainly' or 'of course'. Just answer directly.\n\n"
    "Tool results are injected into the conversation as tool_result messages. You may call "
    "multiple tools in parallel. Each tool call incurs latency, so minimize round-trips.\n\n"
    "Remember: this is an agent pipeline. Clarity and brevity are critical. The downstream "
    "system will parse your output and use it to drive further actions. Any ambiguity or "
    "verbosity in your response will slow down the pipeline.\n\n"
    "You are now ready to receive tasks. Respond to each task below."
)

# Easy prompts — expect short, structured output
EASY_TEMPLATES = [
    'Classify the following review as POSITIVE, NEGATIVE, or NEUTRAL. Reply with only the label.\nReview: "{text}"',
    'Is the following sentence grammatically correct? Answer with only "yes" or "no".\nSentence: "{text}"',
    'Extract the company name from the following text. Reply with only the company name.\nText: "{text}"',
    'Label the following support ticket as BUG, FEATURE_REQUEST, or QUESTION. Reply with only the label.\nTicket: "{text}"',
    'True or false: {claim}. Answer with only "true" or "false".',
    'Is this a valid JSON object? Answer with "valid" or "invalid".\n{text}',
    'Classify the sentiment as POSITIVE or NEGATIVE. One word only.\n"{text}"',
    'Extract the error code from: {text}. Reply with only the error code, nothing else.',
    'Fill in the JSON schema below with values extracted from the context.\nSchema: {{"name": "", "version": "", "language": ""}}\nContext: {text}',
    'Yes or no: does this code contain a syntax error?\n```\n{text}\n```',
]

# Medium prompts — expect a few sentences or a short code snippet
MEDIUM_TEMPLATES = [
    "Summarize the following paragraph in 2-3 sentences:\n\n{text}",
    "Write a one-paragraph explanation of {concept} suitable for a junior developer.",
    "Convert this Python function to JavaScript. Keep it brief:\n```python\n{text}\n```",
    "What are the top 3 differences between {concept_a} and {concept_b}? Be concise.",
    "Explain what this regex does in plain English: {text}",
    "Write a brief docstring for this function:\n```python\n{text}\n```",
    "Translate the following error message into a user-friendly explanation:\n{text}",
    "List 3 common causes of {text} and a one-line fix for each.",
    "Review this short code snippet and identify any obvious issues (2-3 sentences):\n```\n{text}\n```",
    "Write a SQL query to {task}. Keep it simple and correct.",
]

# Hard prompts — expect long, complex output
HARD_TEMPLATES = [
    "Write a function in Python that {task}. Include error handling, type hints, and a docstring.",
    "Implement a {data_structure} from scratch in Python with the following operations: {ops}.",
    "Write a complete REST API endpoint in FastAPI for {task}. Include request/response models and error handling.",
    "Debug the following code and explain all issues found, then provide the corrected version:\n```python\n{text}\n```",
    "Design a step-by-step plan for implementing {feature} in a production system. Cover architecture, data model, API, and testing.",
    "Write unit tests for the following module using pytest:\n```python\n{text}\n```",
    "Implement a multi-step pipeline that: {steps}. Each step should be a separate function.",
    "Write a complete implementation of {algorithm} with O(n log n) complexity. Include analysis.",
]

# Filler content for templates
TEXTS = [
    "The product arrived damaged and customer service was unhelpful.",
    "def add(a, b): return a + b",
    "NullPointerException at line 42 in UserService.java",
    "ERROR_CODE_429: Rate limit exceeded",
    "Acme Corporation released version 3.2.1 of their flagship product last Tuesday.",
    "for i in range(10): print(i*i)",
    '{"name": "requests", "version": "2.28.0", "language": "Python"}',
    "The quick brown fox jumps over the lazy dog. This sentence contains all letters.",
    "TypeError: cannot read property 'length' of undefined",
    "SELECT * FROM users WHERE id = ?",
]

CONCEPTS = [
    "dependency injection", "memoization", "event sourcing",
    "CQRS", "CAP theorem", "consistent hashing",
]

CONCEPT_PAIRS = [
    ("REST", "GraphQL"), ("SQL", "NoSQL"), ("TCP", "UDP"),
    ("Docker", "Kubernetes"), ("async", "threading"),
]

TASKS = [
    "fetch paginated results from a database",
    "validate and sanitize user input",
    "implement retry logic with exponential backoff",
    "parse a CSV file and compute column statistics",
    "authenticate users via JWT tokens",
]

DATA_STRUCTURES = ["binary search tree", "LRU cache", "min-heap", "trie", "union-find"]
OPS_LIST = [
    "insert, delete, search",
    "push, pop, peek, is_empty",
    "add, find, union",
]

ALGORITHMS = ["merge sort", "Dijkstra's shortest path", "quickselect", "Boyer-Moore voting"]

FEATURES = [
    "real-time notifications",
    "multi-tenant data isolation",
    "distributed rate limiting",
    "full-text search",
]

PIPELINE_STEPS = [
    "fetch data from API, parse JSON, validate schema, store in database",
    "read CSV, clean missing values, compute statistics, write report",
]


def build_easy_prompt() -> tuple[str, str]:
    template = random.choice(EASY_TEMPLATES)
    prompt = template.format(
        text=random.choice(TEXTS),
        claim=f"Python lists are O(1) for append",
    )
    full = f"{SYSTEM_PROMPT}\n\n{prompt}"
    return full, "classify/extract/yes-no"


def build_medium_prompt() -> tuple[str, str]:
    template = random.choice(MEDIUM_TEMPLATES)
    ca, cb = random.choice(CONCEPT_PAIRS)
    prompt = template.format(
        text=random.choice(TEXTS),
        concept=random.choice(CONCEPTS),
        concept_a=ca,
        concept_b=cb,
        task=random.choice(TASKS),
    )
    full = f"{SYSTEM_PROMPT}\n\n{prompt}"
    return full, "summarize/short-code"


def build_hard_prompt() -> tuple[str, str]:
    template = random.choice(HARD_TEMPLATES)
    prompt = template.format(
        text=random.choice(TEXTS),
        task=random.choice(TASKS),
        data_structure=random.choice(DATA_STRUCTURES),
        ops=random.choice(OPS_LIST),
        algorithm=random.choice(ALGORITHMS),
        feature=random.choice(FEATURES),
        steps=random.choice(PIPELINE_STEPS),
    )
    full = f"{SYSTEM_PROMPT}\n\n{prompt}"
    return full, "long-code/planning"


def generate_trace(n: int = 50, seed: int = 42) -> list[dict]:
    random.seed(seed)

    # Target distribution
    n_easy   = int(n * 0.60)
    n_medium = int(n * 0.25)
    n_hard   = n - n_easy - n_medium  # ~15%

    requests = []
    difficulty_labels = (
        ["easy"]   * n_easy +
        ["medium"] * n_medium +
        ["hard"]   * n_hard
    )
    random.shuffle(difficulty_labels)

    # Generate prompts
    prompts = []
    for diff in difficulty_labels:
        if diff == "easy":
            prompt, cat = build_easy_prompt()
        elif diff == "medium":
            prompt, cat = build_medium_prompt()
        else:
            prompt, cat = build_hard_prompt()
        prompts.append((prompt, cat, diff))

    # Assign burst-style arrival times: groups of 5-8 with 200-800ms between bursts
    arrival_ms = 0.0
    idx = 0
    request_id = 0
    while idx < n:
        burst_size = random.randint(5, 8)
        for j in range(burst_size):
            if idx >= n:
                break
            prompt, category, diff = prompts[idx]
            # Small jitter within a burst (0-50ms)
            jitter_ms = random.uniform(0, 50)
            requests.append({
                "request_id": f"req_{request_id:03d}",
                "prompt": prompt,
                "arrival_delay_ms": round(arrival_ms + jitter_ms, 1),
                "category": category,
                "expected_difficulty": diff,
            })
            idx += 1
            request_id += 1
        # Pause between bursts: 200-800ms
        arrival_ms += random.uniform(200, 800)

    return requests


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    trace = generate_trace(n=50)
    with open(OUTPUT_PATH, "w") as f:
        for record in trace:
            f.write(json.dumps(record) + "\n")
    print(f"Generated {len(trace)} requests → {OUTPUT_PATH}")
    counts = {}
    for r in trace:
        d = r["expected_difficulty"]
        counts[d] = counts.get(d, 0) + 1
    for d, c in sorted(counts.items()):
        print(f"  {d}: {c} ({100*c//len(trace)}%)")


if __name__ == "__main__":
    main()
