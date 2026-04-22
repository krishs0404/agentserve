"""
OpenAI-compatible FastAPI server for AgentServe.

Endpoints:
  POST /v1/chat/completions  — OpenAI chat completions API
  GET  /healthz              — health check
  GET  /stats                — engine metrics
  GET  /v1/models            — list available models (static)
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from agentserve.model.config import TinyConfig, Llama32_1B, Llama32_3B, Llama32_8B
from agentserve.engine.engine import Engine
from agentserve.engine.request import Request

# ---------------------------------------------------------------------------
# Request / response schemas (subset of OpenAI API)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "agentserve"
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict


# ---------------------------------------------------------------------------
# Global engine (initialised at startup)
# ---------------------------------------------------------------------------

_engine: Optional[Engine] = None
_tokenizer = None


def _get_engine() -> Engine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    return _engine


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Concatenate chat messages into a flat prompt string."""
    parts = []
    for msg in messages:
        parts.append(f"<|{msg.role}|>\n{msg.content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _prompt_to_tokens(prompt: str) -> List[int]:
    """Tokenise a prompt. Falls back to byte-level encoding if no tokenizer."""
    if _tokenizer is not None:
        return _tokenizer.encode(prompt)
    # Simple fallback: map each character to its ASCII value, clamped to vocab
    cfg = _engine.config if _engine else TinyConfig
    return [ord(c) % cfg.vocab_size for c in prompt]


def _tokens_to_text(token_ids: List[int]) -> str:
    """Decode token IDs to text."""
    if _tokenizer is not None:
        return _tokenizer.decode(token_ids, skip_special_tokens=True)
    return "".join(chr(max(32, t % 127)) for t in token_ids)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Initialise engine on startup, clean up on shutdown."""
    global _engine, _tokenizer

    model_name = os.environ.get("AGENTSERVE_MODEL", "mock")
    use_mock = model_name in ("mock", "tiny", "")
    agent_aware = os.environ.get("AGENTSERVE_BASELINE", "0") != "1"

    if use_mock:
        config = TinyConfig
    elif "1b" in model_name.lower():
        config = Llama32_1B
    elif "3b" in model_name.lower():
        config = Llama32_3B
    elif "8b" in model_name.lower():
        config = Llama32_8B
    else:
        config = TinyConfig

    _engine = Engine(
        config=config,
        use_mock=use_mock,
        agent_aware=agent_aware,
        max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "8")),
    )

    if not use_mock:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            pass  # run without tokenizer

    yield  # app is running

    # Cleanup (model references released on GC)
    _engine = None
    _tokenizer = None


app = FastAPI(title="AgentServe", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "engine_ready": _engine is not None}


@app.get("/stats")
async def stats():
    engine = _get_engine()
    m = engine.metrics
    return {
        "total_requests": m.total_requests,
        "completed_requests": m.completed_requests,
        "total_prompt_tokens": m.total_prompt_tokens,
        "total_output_tokens": m.total_output_tokens,
        "throughput_tokens_per_sec": m.throughput_tokens_per_sec,
        "prefix_cache_hit_rate": m.prefix_hit_rate,
        "prefix_tokens_saved": m.prefix_tokens_saved,
        "difficulty_breakdown": m.difficulty_counts,
        "steps": m.steps,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "agentserve", "object": "model", "created": int(time.time())}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    engine = _get_engine()

    prompt = _messages_to_prompt(req.messages)
    token_ids = _prompt_to_tokens(prompt)

    agentserve_req = Request(
        prompt=prompt,
        token_ids=token_ids,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    )
    engine.submit(agentserve_req)

    # Run engine steps until this request completes
    max_steps = req.max_tokens + len(token_ids) + 100
    for _ in range(max_steps):
        if agentserve_req.is_done:
            break
        await asyncio.to_thread(engine.step)

    output_text = _tokens_to_text(agentserve_req.output_token_ids)

    return ChatCompletionResponse(
        id=f"chatcmpl-{agentserve_req.request_id}",
        created=int(time.time()),
        model=req.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": output_text},
            "finish_reason": "stop" if agentserve_req.is_done else "length",
        }],
        usage={
            "prompt_tokens": agentserve_req.num_prompt_tokens,
            "completion_tokens": agentserve_req.num_output_tokens,
            "total_tokens": agentserve_req.num_prompt_tokens + agentserve_req.num_output_tokens,
        },
    )
