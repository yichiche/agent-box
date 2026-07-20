#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

KEY_FILE="${HOST_HOME}/.codex_api_key"

if [ -f "$KEY_FILE" ]; then
  CODEX_KEY=$(cat "$KEY_FILE")
else
  echo ""
  echo "============================================"
  echo "  Codex setup: AMD LLM Gateway key required"
  echo "============================================"
  read -rp "Enter your AMD LLM Gateway API key: " CODEX_KEY
  if [ -z "$CODEX_KEY" ]; then
    echo "[codex] Error: API key cannot be empty."
    exit 1
  fi
  echo "$CODEX_KEY" > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  echo "[codex] API key saved to $KEY_FILE"
fi

apt-get update
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt install nodejs -y

# Install Codex CLI (pinned to 0.132.0 — newer breaks AMD Gateway)
if ! command -v codex >/dev/null 2>&1; then
  echo "[codex] Installing @openai/codex@0.132.0..."
  npm install -g @openai/codex@0.132.0 2>&1 | tail -1
fi

# Write proxy script
mkdir -p /root/.codex
cat > /root/.codex/amd-gateway-proxy.py <<'PYEOF'
#!/usr/bin/env python3
"""Reverse proxy: Bearer auth -> Ocp-Apim-Subscription-Key, path rewrite for AMD LLM Gateway.

Supports SSE streaming (chunked transfer encoding) for the Responses API.
Rejects WebSocket upgrade attempts cleanly so Codex falls back without a scary warning.
"""

import http.server
import http.client
import json
import ssl
import urllib.parse

UPSTREAM_HOST = "llm-api.amd.com"
UPSTREAM_BASE = "/OpenAI"
LISTEN_PORT = 18742


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def _is_websocket_upgrade(self):
        upgrade = self.headers.get("Upgrade", "").lower()
        return upgrade == "websocket"

    def _reject_websocket(self):
        body = json.dumps({"error": "WebSocket not supported, use HTTPS"}).encode()
        self.send_response(426)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _build_headers(self):
        headers = {}
        for key, val in self.headers.items():
            k = key.lower()
            if k in ("host", "transfer-encoding", "upgrade", "connection"):
                continue
            if k == "authorization" and val.lower().startswith("bearer "):
                api_key = val.split(" ", 1)[1]
                headers["Ocp-Apim-Subscription-Key"] = api_key
                continue
            headers[key] = val
        return headers

    def _is_streaming_request(self, body):
        if not body:
            return False
        try:
            data = json.loads(body)
            return data.get("stream", False)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False

    def do_request(self):
        if self._is_websocket_upgrade():
            self._reject_websocket()
            return

        path = self.path
        if path.startswith("/v1"):
            path = path[3:]
        upstream_path = UPSTREAM_BASE + path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else None

        headers = self._build_headers()
        is_streaming = self._is_streaming_request(body)

        ctx = ssl.create_default_context()
        conn = http.client.HTTPSConnection(UPSTREAM_HOST, timeout=300, context=ctx)
        try:
            conn.request(self.command, upstream_path, body=body, headers=headers)
            resp = conn.getresponse()

            self.send_response(resp.status)
            for key, val in resp.getheaders():
                k = key.lower()
                if k in ("transfer-encoding", "connection", "content-length"):
                    continue
                self.send_header(key, val)

            if is_streaming and resp.status == 200:
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(f"{len(chunk):x}\r\n".encode())
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                resp_body = resp.read()
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except BrokenPipeError:
            pass
        except Exception as e:
            error_body = json.dumps({"error": str(e)}).encode()
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(error_body)))
                self.end_headers()
                self.wfile.write(error_body)
            except Exception:
                pass
        finally:
            conn.close()

    do_GET = do_request
    do_POST = do_request
    do_PUT = do_request
    do_DELETE = do_request
    do_PATCH = do_request
    do_OPTIONS = do_request


class ReusableHTTPServer(http.server.HTTPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    server = ReusableHTTPServer(("127.0.0.1", LISTEN_PORT), ProxyHandler)
    server.serve_forever()
PYEOF

# Write Codex auth
cat > /root/.codex/auth.json <<EOF
{
  "auth_mode": "apikey",
  "OPENAI_API_KEY": "${CODEX_KEY}",
  "tokens": null,
  "last_refresh": null
}
EOF
chmod 600 /root/.codex/auth.json

# Write Codex config (preserve existing project trust levels)
if [ ! -f /root/.codex/config.toml ] || ! grep -q "openai_base_url" /root/.codex/config.toml 2>/dev/null; then
  cat > /root/.codex/config.toml <<'EOF'
model = "gpt-5.6-sol"
openai_base_url = "http://127.0.0.1:18742/v1"

[projects."/sgl-workspace"]
trust_level = "untrusted"

[projects."/sgl-workspace/sglang"]
trust_level = "trusted"

[projects."/home/yichiche/agent-box"]
trust_level = "trusted"

[projects."/sgl-workspace/aiter"]
trust_level = "trusted"
EOF
fi

# Keep existing installs on the intended default model without rewriting
# project trust settings or other user-managed config.
if grep -q '^model = ' /root/.codex/config.toml 2>/dev/null; then
  sed -i 's/^model = .*/model = "gpt-5.6-sol"/' /root/.codex/config.toml
else
  sed -i '1imodel = "gpt-5.6-sol"' /root/.codex/config.toml
fi

# Codex only reads openai_base_url at top level — a copy under [profiles.gateway]
# is ignored and Codex falls back to api.openai.com (401 with AMD gateway keys).
if grep -q '^openai_base_url = ' /root/.codex/config.toml 2>/dev/null; then
  sed -i 's|^openai_base_url = .*|openai_base_url = "http://127.0.0.1:18742/v1"|' /root/.codex/config.toml
else
  sed -i '1iopenai_base_url = "http://127.0.0.1:18742/v1"' /root/.codex/config.toml
fi

# Append env vars and proxy auto-start to .bashrc (idempotent)
if ! grep -q "codex-proxy" /root/.bashrc 2>/dev/null; then
  cat >> /root/.bashrc <<EOF

# >>> codex-proxy (agent-box managed) >>>
export OPENAI_API_KEY="${CODEX_KEY}"
if ! pgrep -f "amd-gateway-proxy" >/dev/null 2>&1; then
  nohup python3 ~/.codex/amd-gateway-proxy.py > ~/.codex/proxy.log 2>&1 &
fi
# <<< codex-proxy (agent-box managed) <<<
EOF
fi

# Start proxy now
if ! pgrep -f "amd-gateway-proxy" >/dev/null 2>&1; then
  nohup python3 /root/.codex/amd-gateway-proxy.py > /root/.codex/proxy.log 2>&1 &
fi
export OPENAI_API_KEY="${CODEX_KEY}"

echo "[codex] Setup complete (proxy on :18742, model: gpt-5.6-sol)"
