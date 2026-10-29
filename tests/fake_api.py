"""A local stand-in for an embeddings + chat API, so tests run offline and free.

Embeddings are hashed bags of words, which is crude but enough for word-overlap retrieval.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 64


def embed(text: str) -> list[float]:
    v = [0.0] * DIM
    for word in re.findall(r"[a-z]+", text.lower()):
        if len(word) > 3:
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    return v


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            data = [{"index": i, "embedding": embed(t)} for i, t in enumerate(body["input"])]
            out = {"data": data, "usage": {"total_tokens": sum(len(t.split()) for t in body["input"])}}
        else:
            prompt = body["messages"][-1]["content"]
            reply = "yes" if prompt.rstrip().endswith("yes or no.") else prompt.split("[1] ", 1)[1].split("\n")[0]
            if self.path.endswith("/v1/messages"):
                out = {"content": [{"type": "text", "text": reply}], "usage": {"input_tokens": 10, "output_tokens": 2}}
            else:
                out = {"choices": [{"message": {"content": reply}}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def start() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"
