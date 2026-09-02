FROM python:3.11-slim

# Unbuffered stdout/stderr so `docker logs` shows output promptly (Python
# fully-buffers stdout when it's not a TTY, which a container's is not).
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Build context is the repo root: docker build -f src/gepa/rpc/Dockerfile .
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && pip install --no-cache-dir ".[rpc]" \
 && apt-get purge -y --auto-remove git \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --no-create-home --shell /bin/false appuser && chown appuser /app
USER appuser

EXPOSE 50051

ENTRYPOINT ["gepa-rpc", "--port", "50051"]
