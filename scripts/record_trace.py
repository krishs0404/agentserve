"""
Utility to record an agent's API calls into a replayable trace file.

Run a proxy server that intercepts OpenAI-compatible API calls and logs them
to a JSONL file. The trace can then be replayed through AgentServe via
bench_agent_trace.py.

Usage:
  # Start the recorder (intercepts calls to target URL):
  uv run python scripts/record_trace.py \
      --target http://localhost:8000 \
      --output traces/recorded_agent.jsonl \
      --port 9000

  # Then point your agent at http://localhost:9000 instead of the real API.
"""

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen, Request as URLRequest
import threading


class TraceRecorder:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.start_time = time.monotonic()
        self.lock = threading.Lock()
        self.count = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def record(self, prompt: str, messages: list, response_text: str) -> None:
        arrival_ms = (time.monotonic() - self.start_time) * 1000
        with self.lock:
            self.count += 1
            record = {
                "request_id": f"recorded_{self.count:04d}",
                "prompt": prompt,
                "arrival_delay_ms": round(arrival_ms, 1),
                "category": "recorded",
                "expected_difficulty": "unknown",
                "messages": messages,
                "response_text": response_text,
            }
            with open(self.output_path, "a") as f:
                f.write(json.dumps(record) + "\n")


_recorder: TraceRecorder | None = None
_target_url: str = ""


class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)

        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        # Forward to target
        target = _target_url.rstrip("/") + self.path
        try:
            req = URLRequest(target, data=body, headers=dict(self.headers), method="POST")
            with urlopen(req, timeout=60) as resp:
                resp_body = resp.read()
                status = resp.status
                resp_headers = dict(resp.headers)
        except Exception as e:
            self.send_error(502, str(e))
            return

        # Record the call
        if _recorder and "/chat/completions" in self.path:
            messages = payload.get("messages", [])
            prompt = " ".join(m.get("content", "") for m in messages)
            try:
                resp_json = json.loads(resp_body)
                response_text = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception:
                response_text = ""
            _recorder.record(prompt, messages, response_text)

        self.send_response(status)
        for k, v in resp_headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(resp_body)

    def log_message(self, fmt, *args):
        pass  # suppress default logging


def main():
    global _recorder, _target_url

    parser = argparse.ArgumentParser(description="Record agent API calls to a trace file")
    parser.add_argument("--target", default="http://localhost:8000", help="Target API URL to proxy")
    parser.add_argument("--output", default="traces/recorded_agent.jsonl")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    output = os.path.join(os.path.dirname(__file__), "..", args.output)
    _recorder = TraceRecorder(output)
    _target_url = args.target

    server = HTTPServer(("0.0.0.0", args.port), ProxyHandler)
    print(f"Recording proxy listening on :{args.port}")
    print(f"Forwarding to: {args.target}")
    print(f"Saving trace to: {output}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"\nRecorded {_recorder.count} requests to {output}")


if __name__ == "__main__":
    main()
