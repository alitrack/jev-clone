#!/usr/bin/env python3
"""Role-rewriting HTTP shim in front of a strict OpenAI-compatible endpoint.

Why this exists
---------------
SGLang (115:8014) enforces "System message must be at the beginning" and answers
HTTP 400 when any `role: "system"` message appears at an index other than 0.
Agent harnesses legitimately place extra system messages mid-conversation:

  * sigil injects its goals block as a system message at index 1
    (`inject_goals_block`, src/agent/mod.rs:381-396), and
  * sigil records context compaction as a mid-list system message starting with
    "[earlier conversation compressed]" (src/agent/mod.rs:240).

Either one makes a long sigil run die with
    provider error 400: {"message":"System message must be at the beginning."}
after minutes of work, because the harness never reaches the point of writing
files. Rewriting the offending messages to `user` (with a `[system] ` prefix so
nothing is lost) is a one-line-per-message fix that keeps the local 27B usable
without touching either project.

Usage
-----
    python3 tools/sysrole-proxy.py                 # listen 127.0.0.1:8090
    LISTEN=127.0.0.1:8091 UP_PORT=8014 python3 tools/sysrole-proxy.py

Then point the client at http://127.0.0.1:8090/v1.

Everything is passed through untouched except the message roles of request
bodies: paths, headers, query strings, status codes and streaming framing
(chunked pass-through) are preserved verbatim.

Stdlib only — no aiohttp/uvicorn needed.
"""

import http.client
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UP_HOST = os.environ.get("UP_HOST", "127.0.0.1")
UP_PORT = int(os.environ.get("UP_PORT", "8014"))
LISTEN_HOST, _, LISTEN_PORT = os.environ.get("LISTEN", "127.0.0.1:8090").partition(":")
LOG_PATH = os.environ.get("PROXY_LOG", "/mnt/d/wsl2/tmp/jev-m0/proxy.log")

STATS = {"requests": 0, "rewrites": 0}
LOCK = threading.Lock()


def log(line: str) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line.rstrip("\n") + "\n")
    except OSError:
        pass


def rewrite_system_roles(raw: bytes) -> tuple[bytes, str]:
    """Move every non-leading `system` message to `user`. Returns (body, note)."""
    if not raw:
        return raw, ""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return raw, ""
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        return raw, ""

    fixed = []
    for i, m in enumerate(msgs):
        if isinstance(m, dict) and m.get("role") == "system" and i != 0:
            m["role"] = "user"
            content = m.get("content")
            if isinstance(content, str):
                m["content"] = "[system] " + content
            fixed.append(i)

    if not fixed:
        return raw, ""
    with LOCK:
        STATS["rewrites"] += len(fixed)
    return json.dumps(obj).encode("utf-8"), f"rewrote system->user at idx {fixed}"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "sysrole-proxy/1.0"

    def log_message(self, format, *args):  # keep the console readable
        pass

    def _proxy(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""

        note = ""
        if body:
            body, note = rewrite_system_roles(body)

        with LOCK:
            STATS["requests"] += 1
            n = STATS["requests"]
        n_msgs = ""
        if body:
            try:
                n_msgs = f" msgs={len(json.loads(body).get('messages', []))}"
            except Exception:  # noqa: BLE001
                pass
        log(f"#{n} {method} {self.path}{n_msgs} {('| ' + note) if note else ''}")

        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length", "connection", "accept-encoding")
        }
        headers["Content-Length"] = str(len(body))

        try:
            conn = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=3600)
            conn.request(method, self.path, body=body or None, headers=headers)
            resp = conn.getresponse()
        except Exception as exc:  # noqa: BLE001
            log(f"#{n} !! upstream error: {exc}")
            self.send_response(502)
            payload = json.dumps({"error": {"message": f"proxy upstream error: {exc}"}}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        chunked = "chunked" in (resp.getheader("Transfer-Encoding") or "").lower()
        body_full = b""
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() in ("transfer-encoding", "content-length", "connection"):
                continue
            self.send_header(k, v)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            body_full = resp.read()
            self.send_header("Content-Length", str(len(body_full)))
        self.end_headers()

        try:
            if chunked:
                # resp.fp yields the raw (still chunk-framed) bytes -> pass through
                while True:
                    chunk = resp.fp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            else:
                self.wfile.write(body_full)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log(f"#{n} client disconnected mid-stream")
        finally:
            conn.close()
            if resp.status != 200:
                log(f"#{n} upstream status {resp.status}")

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")


def main() -> int:
    addr = (LISTEN_HOST, int(LISTEN_PORT))
    log(f"--- sysrole-proxy listening on {LISTEN_HOST}:{LISTEN_PORT} -> {UP_HOST}:{UP_PORT}")
    srv = ThreadingHTTPServer(addr, Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    log(f"--- sysrole-proxy stopped (requests={STATS['requests']} rewrites={STATS['rewrites']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
