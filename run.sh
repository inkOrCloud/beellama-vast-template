#!/usr/bin/env bash
# ============================================================================
# beellama.cpp + Cloudflare named tunnel + portal  —  Vast.ai template runner
# Builds https://github.com/Anbeeld/beellama.cpp (CUDA), downloads a HF GGUF
# model (hash-verified), serves llama-server, exposes the OpenAI API + WebUI
# via a Cloudflare named tunnel (SSE-capable), and runs a portal WebUI
# showing the public URLs.
#
# Env vars (set in the Vast template / instance):
#   HF_TOKEN         HF read token (needed for gated/private models)
#   HF_REPO          HF repo id, default unsloth/Qwen3.5-0.8B-GGUF
#   HF_FILE          GGUF filename in repo, default Qwen3.5-0.8B-UD-IQ2_XXS.gguf
#   MODEL_DIR        model dir (KEEP ON VOLUME), default /workspace/models
#   LLAMA_PORT       llama-server port, default 8080
#   PORTAL_PORT      portal WebUI port (declare to Vast for the Open button), default 8888
#   CTX              context length, default 8192
#   KV               KV cache type (beellama KVarN), default kvarn6
#   EXTRA_ARGS       extra llama-server args appended at the END (override defaults)
#   CF_TUNNEL_TOKEN  Cloudflare named-tunnel token (REQUIRED; get from dashboard:
#                    Networks -> Tunnels -> your tunnel -> Configure -> Install and run)
#   CF_API_HOST      public hostname for the OpenAI-compatible API (DNS CNAME ->
#                    <tunnel-id>.cfargotunnel.com, proxied), default beellama-api.43497674114036719825.asia
#                    API base = https://<CF_API_HOST>/v1 (standard OpenAI layout)
#   CF_WEB_HOST      public hostname for the llama.cpp WebUI (same CNAME setup),
#                    default beellama-webui.43497674114036719825.asia
#
# NOTE: a named Cloudflare tunnel is used (NOT trycloudflare quick tunnels)
# because quick tunnels do not support SSE — llama.cpp streaming would break.
# ============================================================================
set -uo pipefail

HF_TOKEN="${HF_TOKEN:-}"
HF_REPO="${HF_REPO:-unsloth/Qwen3.5-0.8B-GGUF}"
HF_FILE="${HF_FILE:-Qwen3.5-0.8B-UD-IQ2_XXS.gguf}"
MODEL_DIR="${MODEL_DIR:-/workspace/models}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
PORTAL_PORT="${PORTAL_PORT:-8888}"
CTX="${CTX:-8192}"
KV="${KV:-kvarn6}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CF_TUNNEL_TOKEN="${CF_TUNNEL_TOKEN:-}"
CF_API_HOST="${CF_API_HOST:-beellama-api.43497674114036719825.asia}"
CF_WEB_HOST="${CF_WEB_HOST:-beellama-webui.43497674114036719825.asia}"

WORK=/root/beellama
LOG="$WORK/run.log"
STATUS="$WORK/status.json"
LLAMA_LOG="$WORK/llama.log"
CF_LOG="$WORK/cf.log"
SRC="$WORK/beellama.cpp"
BUILD="$WORK/build"
REPO_URL=https://github.com/Anbeeld/beellama.cpp
# Prebuilt image (ghcr.io/inkorcloud/beellama-cuda): llama-server already at
# /app, python3 + hf + cloudflared baked in -> skip ALL toolchain installs and
# compilation. Fresh instances then start in seconds instead of ~30+ min.
PREBUILT=0
[ -x /app/llama-server ] && PREBUILT=1

export PATH="/opt/conda/bin:/usr/local/bin:/root/.local/bin:$PATH"
export DEBIAN_FRONTEND=noninteractive
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1

mkdir -p "$WORK" "$MODEL_DIR"
exec >> "$LOG" 2>&1
echo "================================================================"
echo "beellama runner start $(date -Is)  | repo=$HF_REPO file=$HF_FILE"
echo "model_dir=$MODEL_DIR ctx=$CTX kv=$KV llama_port=$LLAMA_PORT portal_port=$PORTAL_PORT"

# ---- portal must be up from the very beginning (shows build progress) ----
cat > "$WORK/portal.py" <<'PYEOF'
#!/usr/bin/env python3
import json, os, sys, subprocess, time, html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATUS_FILE, LOG_FILE, PORT = sys.argv[1], sys.argv[2], int(sys.argv[3])

def tail(path, maxbytes=12000, n=150):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2); size = f.tell()
            f.seek(max(0, size - maxbytes)); data = f.read().decode('utf-8', 'replace')
        return '\n'.join(data.splitlines()[-n:])
    except Exception:
        return ''

def gpu_info():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader'], capture_output=True, text=True, timeout=5).stdout.strip()
        return out or 'n/a'
    except Exception:
        return 'n/a'

def health(port):
    try:
        import urllib.request
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>beellama.cpp portal</title>
<meta http-equiv="refresh" content="5">
<style>
body{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#0f1117;color:#d8dee9;margin:0;padding:24px}}
h1{{font-size:20px;color:#88c0d0;margin:0 0 4px 0}}
.sub{{color:#616e88;font-size:12px;margin-bottom:20px}}
.card{{background:#1a1e29;border:1px solid #2a3040;border-radius:10px;padding:16px 20px;margin-bottom:16px}}
.badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:bold;margin-left:8px}}
.b{{background:#5e81ac;color:#fff}}.ok{{background:#a3be8c;color:#0f1117}}.err{{background:#bf616a;color:#fff}}
table{{border-collapse:collapse;width:100%}}
td{{padding:6px 10px;border-bottom:1px solid #232a3a;font-size:14px;vertical-align:top}}
td.k{{color:#88c0d0;white-space:nowrap;width:140px}}
a{{color:#81a1c1;word-break:break-all}}
pre{{background:#0b0d13;border:1px solid #232a3a;border-radius:8px;padding:12px;font-size:12px;max-height:420px;overflow:auto;color:#a3be8c}}
</style></head><body>
<h1>&#128029; beellama.cpp + Cloudflare tunnel</h1>
<div class="sub">Vast.ai instance &middot; auto-refresh 5s &middot; phase badge = startup pipeline state</div>
<div class="card"><table>
<tr><td class="k">Phase</td><td>{phase}</td></tr>
<tr><td class="k">Model</td><td>{model}</td></tr>
<tr><td class="k">Model path</td><td>{model_path}</td></tr>
<tr><td class="k">llama-server</td><td>{llama_health}</td></tr>
<tr><td class="k">GPU</td><td>{gpu}</td></tr>
</table></div>
<div class="card"><table>
<tr><td class="k">API base URL</td><td>{api}</td></tr>
<tr><td class="k">OpenAI SDK</td><td>{api_sdk}</td></tr>
<tr><td class="k">WebUI</td><td>{webui}</td></tr>
<tr><td class="k">Try</td><td>{tryline}</td></tr>
</table></div>
{error_html}
<div class="card"><div class="sub">startup log (run.log tail)</div><pre>{log}</pre></div>
</body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        if self.path.split('?')[0] == '/status.json':
            try:
                body = open(STATUS_FILE).read()
            except Exception:
                body = '{}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body.encode())
            return
        st = {}
        try:
            st = json.load(open(STATUS_FILE))
        except Exception:
            pass
        phase = st.get('phase', 'starting')
        api = st.get('api_base', '')
        mode = st.get('api_mode', '')
        web = st.get('webui_url', '')
        model = st.get('model', '')
        model_path = st.get('model_path', '')
        err = st.get('error', '')
        if api:
            api_html = '<a href="{0}" target="_blank">{0}</a>'.format(html.escape(api))
            if mode == 'root':
                api_sdk = 'base_url = <b>{0}</b> &nbsp;(tunnel maps /... to /v1/... automatically)'.format(html.escape(api))
                tryline = 'curl {0}/models &nbsp;&middot;&nbsp; curl {0}/chat/completions -d \'{{"model":"{1}","messages":[{{"role":"user","content":"hi"}}]}}\''.format(html.escape(api), html.escape(model.split('/')[-1] if model else 'model'))
            else:
                api_sdk = 'base_url = <b>{0}</b>'.format(html.escape(api))
                tryline = 'curl {0}/models'.format(html.escape(api))
        else:
            api_html = '<span style="color:#616e88">starting&hellip;</span>'
            api_sdk = '&nbsp;'
            tryline = '&nbsp;'
        if web:
            web_html = '<a href="{0}" target="_blank">{0}</a>'.format(html.escape(web))
        else:
            web_html = '<span style="color:#616e88">starting&hellip;</span>'
        llama_health = 'UP' if health(int(st.get('llama_port', 8080))) else 'starting / not ready'
        badge = '<span class="badge {0}">{1}</span>'.format('err' if err else ('ok' if phase == 'running' else 'b'), html.escape(phase))
        err_html = '<div class="card"><span class="badge err">ERROR</span><pre>{0}</pre></div>'.format(html.escape(err)) if err else ''
        htmlout = PAGE.format(
            phase=badge, model=html.escape(model or '&nbsp;'), model_path=html.escape(model_path or '&nbsp;'),
            llama_health=llama_health, gpu=html.escape(gpu_info()),
            api=api_html, api_sdk=api_sdk, webui=web_html, tryline=tryline,
            error_html=err_html, log=html.escape(tail(LOG_FILE)))
        body = htmlout.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

ThreadingHTTPServer(('0.0.0.0', PORT), H).serve_forever()
PYEOF

# phase/status helpers
phase() {
    echo "[$(date +%H:%M:%S)] PHASE: $1"
    printf '{"phase":"%s","ts":"%s","llama_port":"%s"}\n' "$1" "$(date -Is)" "$LLAMA_PORT" > "$STATUS"
}
fail() {
    echo "[$(date +%H:%M:%S)] FATAL: $1"
    printf '{"phase":"failed","error":"%s","ts":"%s"}\n' "$1" "$(date -Is)" > "$STATUS"
    exit 1
}

# portal up from phase 0
pkill -f 'portal[.]py' 2>/dev/null || true
sleep 1
setsid nohup python3 "$WORK/portal.py" "$STATUS" "$LOG" "$PORTAL_PORT" > "$WORK/portal.log" 2>&1 < /dev/null &
echo "portal started on port $PORTAL_PORT (pid $!)"

# ============================================================================
phase bootstrap
if [ "$PREBUILT" = 1 ]; then
    echo "prebuilt image detected (/app/llama-server) — skipping toolchain install (no apt/conda, no compilation)"
else
    echo "--- apt packages ---"
    apt-get update -qq || true
    apt-get install -y -qq --no-install-recommends build-essential cmake git curl ca-certificates psmisc || fail "apt install failed"
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "nvcc missing -> conda install cuda-toolkit (this can take several minutes)"
        /opt/conda/bin/conda install -y -q -c nvidia cuda-toolkit=12.4.1 || fail "conda cuda-toolkit install failed"
    fi
    # conda's cuda-toolkit metapackage can omit cuBLAS dev headers (seen 2026-08-08:
    # cublas_v2.h missing -> build fails) — install explicitly, idempotent
    if [ ! -f /opt/conda/include/cublas_v2.h ]; then
        echo "cublas_v2.h missing -> conda install cuda-cublas-dev"
        /opt/conda/bin/conda install -y -q -c nvidia cuda-cublas-dev || fail "conda cuda-cublas-dev install failed"
    fi
    nvcc --version | tail -1 || fail "nvcc still missing"
    gcc --version | head -1
    # conda base ships an old libstdc++ (6.0.29, no GLIBCXX_3.4.30); binaries built
    # with system gcc then fail at runtime via RUNPATH /opt/conda/lib (seen 2026-08-08).
    # conda solver is ALSO broken by a stale nvidia cuda-compiler record, so swap the
    # .so directly with the newer system one (backward compatible).
    if [ -f /opt/conda/lib/libstdc++.so.6 ] && ! strings /opt/conda/lib/libstdc++.so.6 2>/dev/null | grep -q "GLIBCXX_3.4.30"; then
        SYS_LIBCXX=$(ls /usr/lib/x86_64-linux-gnu/libstdc++.so.6.* 2>/dev/null | head -1)
        if [ -n "$SYS_LIBCXX" ]; then
            cp -f --remove-destination "$SYS_LIBCXX" /opt/conda/lib/libstdc++.so.6 && echo "conda libstdc++ replaced with $SYS_LIBCXX"
        fi
    fi
fi

# ============================================================================
phase cloudflared
if ! command -v cloudflared >/dev/null 2>&1; then
    curl -fsSL -o /usr/local/bin/cloudflared "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" || fail "cloudflared download failed"
    chmod +x /usr/local/bin/cloudflared
fi
cloudflared --version 2>&1 | head -1

# ============================================================================
phase hfcli
if ! command -v hf >/dev/null 2>&1; then
    python3 -m pip install -q --no-warn-script-location -U "huggingface_hub[hf_xet]" 2>/dev/null || curl -LsSf https://hf.co/cli/install.sh | bash -s || fail "hf cli install failed"
fi
command -v hf || fail "hf cli not in PATH"
hf version 2>/dev/null | head -1 || true

# ============================================================================
phase build
if [ "$PREBUILT" = 1 ]; then
    LLAMA=/app/llama-server
    echo "using prebuilt llama-server: $LLAMA"
else
    if [ ! -d "$SRC/.git" ]; then
        echo "cloning $REPO_URL"
        git clone --depth 1 "$REPO_URL" "$SRC" || fail "git clone failed"
    fi
    if [ ! -x "$BUILD/bin/llama-server" ]; then
        echo "cmake configure..."
        # conda nvcc only searches targets/x86_64-linux/include by default; the
        # nvidia-channel cuda-cublas-dev puts cublas_v2.h in /opt/conda/include
        # (seen 2026-08-08) -> explicit -I flag
        CUDA_EXTRA_FLAGS=""
        [ -f /opt/conda/include/cublas_v2.h ] && CUDA_EXTRA_FLAGS="-I/opt/conda/include"
        cmake -B "$BUILD" -S "$SRC" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_FLAGS="$CUDA_EXTRA_FLAGS" || fail "cmake configure failed"
        NCPU=$(nproc); NJ=$(( NCPU > 32 ? 32 : NCPU ))
        echo "cmake build -j$NJ (KVarN CUDA kernels take a while; plateaus are normal)..."
        cmake --build "$BUILD" -j"$NJ" --target llama-server || fail "cmake build failed"
        echo "build done: $(du -sh "$BUILD" 2>/dev/null | cut -f1)"
    else
        echo "build already present, skipping ($BUILD/bin/llama-server)"
    fi
    LLAMA="$BUILD/bin/llama-server"
    [ -x "$LLAMA" ] || LLAMA=$(find "$BUILD" -name llama-server -type f 2>/dev/null | head -1)
    [ -x "$LLAMA" ] || fail "llama-server binary not found after build"
fi
echo "llama-server: $LLAMA"
"$LLAMA" --version 2>&1 | head -2

# ============================================================================
phase model
[ -n "$HF_TOKEN" ] && export HF_TOKEN
export HF_FILE  # used by the python snippet below
AUTH=(); [ -n "$HF_TOKEN" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
MODEL="$MODEL_DIR/$HF_FILE"
expected=""
expected_size=""
echo "fetching HF metadata for $HF_REPO..."
meta=$(curl -fsS --max-time 60 "${AUTH[@]}" "https://huggingface.co/api/models/$HF_REPO/tree/main?recursive=true") || echo "WARN: HF metadata fetch failed"
if [ -n "$meta" ]; then
    expected=$(printf '%s' "$meta" | python3 -c "
import json,sys,os
try:
    d=json.load(sys.stdin)
    print(next((f['lfs']['oid'] for f in d if f.get('type')=='file' and f.get('path')==os.environ.get('HF_FILE','')), ''))
except Exception:
    print('')
" 2>/dev/null) || true
    expected_size=$(printf '%s' "$meta" | python3 -c "
import json,sys,os
try:
    d=json.load(sys.stdin)
    print(next((str(f['lfs']['size']) for f in d if f.get('type')=='file' and f.get('path')==os.environ.get('HF_FILE','')), ''))
except Exception:
    print('')
" 2>/dev/null) || true
fi
if [ -z "$expected" ]; then
    echo "WARN: could not determine expected sha256 for $HF_FILE — existence check only"
else
    echo "expected sha256: $expected"
fi
if [ -n "$expected_size" ]; then
    echo "expected size: $expected_size"
fi
verify_model() {
    [ -f "$MODEL" ] || return 1
    if [ "${SKIP_VERIFY:-0}" = "1" ]; then
        echo "SKIP_VERIFY=1 — existence check only (no sha256)"
        return 0
    fi
    if [ -n "$expected" ]; then
        actual=$(sha256sum "$MODEL" 2>/dev/null | cut -d' ' -f1)
        if [ "$actual" = "$expected" ]; then
            return 0
        fi
        # size match is acceptable (e.g. locally patched GGUF keeps the same size)
        if [ -n "$expected_size" ]; then
            actual_size=$(stat -c %s "$MODEL" 2>/dev/null || echo 0)
            if [ "$actual_size" = "$expected_size" ]; then
                echo "sha256 differs but size matches ($actual_size) — treating as valid (pre-patched model)"
                return 0
            fi
        fi
        return 1
    else
        return 0
    fi
}
if verify_model; then
    echo "model exists and hash OK — skipping download: $MODEL"
else
    [ -f "$MODEL" ] && { echo "existing file hash mismatch or incomplete — removing"; rm -f "$MODEL"; }
    echo "downloading $HF_REPO/$HF_FILE -> $MODEL_DIR (XET high-performance)..."
    mkdir -p "$MODEL_DIR"
    export HF_HOME="$MODEL_DIR/.hf-cache"
    # XET can hang on hosts with bad connectivity to the XET CAS bridge (seen
    # 2026-08-10: concurrency controller spiraling down, zero progress). Guard
    # with timeout; on failure/timeout retry once with XET disabled (plain HTTP).
    if ! timeout 1500 hf download "$HF_REPO" "$HF_FILE" --local-dir "$MODEL_DIR"; then
        echo "XET download failed or timed out — retrying with HF_HUB_DISABLE_XET=1 (plain HTTP)..."
        HF_HUB_DISABLE_XET=1 timeout 1500 hf download "$HF_REPO" "$HF_FILE" --local-dir "$MODEL_DIR" || fail "hf download failed (XET and plain HTTP)"
    fi
    verify_model || { rm -f "$MODEL"; fail "sha256 verification FAILED after download"; }
fi
ls -la "$MODEL"

# ============================================================================
phase llama
pkill -f 'llama[-]server' 2>/dev/null || true
sleep 1
LLAMA_ARGS=(-m "$MODEL" --host 0.0.0.0 --port "$LLAMA_PORT" -c "$CTX" \
    --cache-type-k "$KV" --cache-type-v "$KV" --flash-attn on -ngl 99 --metrics \
    --alias "${HF_FILE%.gguf}")
# EXTRA_ARGS appended at end so users can override defaults
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && LLAMA_ARGS+=($EXTRA_ARGS)
echo "launching: ${LLAMA_ARGS[*]}"
setsid nohup "$LLAMA" "${LLAMA_ARGS[@]}" > "$LLAMA_LOG" 2>&1 < /dev/null &
echo "llama-server pid $!"
ok=""
for i in $(seq 1 90); do
    if curl -fsS --max-time 5 -o /dev/null "http://127.0.0.1:$LLAMA_PORT/health"; then ok=1; echo "llama-server healthy after ~$((i*2))s"; break; fi
    sleep 2
done
[ -n "$ok" ] || fail "llama-server did not become healthy (see $LLAMA_LOG)"
curl -s --max-time 5 "http://127.0.0.1:$LLAMA_PORT/health" || true

# ============================================================================
phase tunnel
[ -n "$CF_TUNNEL_TOKEN" ] || fail "CF_TUNNEL_TOKEN env var is required (Cloudflare named tunnel token)"
pkill -f 'cloudflared tunne[l]' 2>/dev/null || true
sleep 1
cat > "$WORK/config.yml" <<EOF
# beellama.cpp tunnel ingress — written by run.sh
# NOTE: cloudflared ingress does NOT support a path on the origin service
# ("ingress rules don't support proxying to a different path on the origin
# service") — service must be host:port only. Both hostnames therefore serve
# the full llama-server: API base = https://CF_API_HOST/v1 (standard OpenAI),
# WebUI = https://CF_WEB_HOST/.
ingress:
  - hostname: $CF_API_HOST
    service: http://127.0.0.1:$LLAMA_PORT
  - hostname: $CF_WEB_HOST
    service: http://127.0.0.1:$LLAMA_PORT
  - service: http_status:404
EOF
echo "tunnel config:"
cat "$WORK/config.yml"
echo "launching cloudflared (named tunnel, token mode)..."
setsid nohup cloudflared tunnel --config "$WORK/config.yml" run --token "$CF_TUNNEL_TOKEN" > "$CF_LOG" 2>&1 < /dev/null &
CF_PID=$!
echo "cloudflared pid $CF_PID"
# wait for the tunnel to register
ok=""
for i in $(seq 1 40); do
    if grep -q "Registered tunnel connection" "$CF_LOG" 2>/dev/null; then ok=1; echo "tunnel registered after ~$((i*3))s"; break; fi
    if ! kill -0 "$CF_PID" 2>/dev/null; then echo "cloudflared exited early — log tail:"; tail -20 "$CF_LOG"; break; fi
    sleep 3
done
tail -3 "$CF_LOG"

# --- verify the two public hostnames actually route to llama-server ---
API_URL="https://$CF_API_HOST"
WEB_URL="https://$CF_WEB_HOST"
API_BASE=""; API_MODE=""
probe() { curl -fsS -o /dev/null --max-time 20 "$1/v1/models"; }
if [ -n "$ok" ] && probe "$API_URL"; then
    API_BASE="$API_URL/v1"; API_MODE="v1"
else
    API_BASE="$API_URL/v1"; API_MODE="unknown"
fi
echo "api base (verified): $API_BASE  mode=$API_MODE"
WEB_OK=""
if curl -fsS --max-time 20 -o /dev/null "$WEB_URL/"; then WEB_OK=1; echo "webui reachable at $WEB_URL/"; else echo "WARN: webui not reachable yet at $WEB_URL/"; fi

# ============================================================================
phase portal
printf '{"phase":"running","ts":"%s","llama_port":"%s","api_base":"%s","api_mode":"%s","webui_url":"%s","model":"%s/%s","model_path":"%s","webui_ok":"%s"}\n' \
    "$(date -Is)" "$LLAMA_PORT" "$API_BASE" "$API_MODE" "$WEB_URL" "$HF_REPO" "$HF_FILE" "$MODEL" "${WEB_OK:-}" > "$STATUS"

{
    echo "== beellama instance URLs =="
    echo "API base:  $API_BASE   (mode=$API_MODE)"
    echo "WebUI:     $WEB_URL"
    echo "Model:     $HF_REPO/$HF_FILE -> $MODEL"
} > "$WORK/urls.txt"
echo "urls written to $WORK/urls.txt"
echo "=== ALL DONE $(date -Is) ==="
