# Enhanced beellama.cpp CUDA image for Vast.ai templates.
# FROM the official prebuilt beellama server image (no compilation needed at
# instance startup) and bakes in the runtime deps the template runner needs:
# python3 (status portal), huggingface_hub CLI (fast model downloads), and
# cloudflared (named tunnel, SSE-capable). Instance startup then only has to
# download the model + start services.
FROM ghcr.io/anbeeld/beellama.cpp:server-cuda12-v0.4.2

# python3 for the status portal + hf CLI for fast model downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ca-certificates psmisc \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir -q "huggingface_hub[hf_xet]"

# cloudflared named tunnel binary
RUN curl -fsSL -o /usr/local/bin/cloudflared \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    && chmod +x /usr/local/bin/cloudflared

# keep the upstream server entrypoint (Vast runs its own supervisor anyway)
ENTRYPOINT ["/app/llama-server"]
