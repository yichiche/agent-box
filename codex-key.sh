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
"""Reverse proxy: Bearer auth -> Ocp-Apim-Subscription-Key, path rewrite for AMD LLM Gateway."""

import http.server
import urllib.request
import urllib.error

UPSTREAM = "https://llm-api.amd.com/OpenAI"
LISTEN_PORT = 18741


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def do_request(self):
        path = self.path
        if path.startswith("/v1"):
            path = path[3:]
        upstream_url = UPSTREAM.rstrip("/") + path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else None

        headers = {}
        for key, val in self.headers.items():
            k = key.lower()
            if k in ("host", "transfer-encoding"):
                continue
            if k == "authorization" and val.lower().startswith("bearer "):
                api_key = val.split(" ", 1)[1]
                headers["Ocp-Apim-Subscription-Key"] = api_key
                continue
            headers[key] = val

        req = urllib.request.Request(upstream_url, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for key, val in resp.getheaders():
                    if key.lower() not in ("transfer-encoding", "connection", "content-length"):
                        self.send_header(key, val)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            resp_body = e.read()
            self.send_response(e.code)
            for key, val in e.headers.items():
                if key.lower() not in ("transfer-encoding", "connection", "content-length"):
                    self.send_header(key, val)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

    do_GET = do_request
    do_POST = do_request
    do_PUT = do_request
    do_DELETE = do_request
    do_PATCH = do_request
    do_OPTIONS = do_request


if __name__ == "__main__":
    server = http.server.HTTPServer(("127.0.0.1", LISTEN_PORT), ProxyHandler)
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
model = "gpt-4.5"
openai_base_url = "http://127.0.0.1:18741/v1"

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

echo "[codex] Setup complete (proxy on :18741, model: gpt-4.5)"
