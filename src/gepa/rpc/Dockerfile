# Digest-pinned so a rebuild can't silently pick up a different base image.
# Refresh with: docker buildx imagetools inspect python:3.11-slim
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

# Unbuffered stdout/stderr so `docker logs` shows output promptly (Python
# fully-buffers stdout when it's not a TTY, which a container's is not).
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Build context is the repo root: docker build -f src/gepa/rpc/Dockerfile .
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# uv.lock was previously copied in but never used -- `pip install ".[rpc]"`
# re-resolves from pyproject.toml, so image contents drifted from the locked
# versions CI tests against. Export the locked set and install exactly that.
#
# --no-default-groups and --no-dev keep this production-only: the project has
# no PEP 735 dependency-groups today, but being explicit means adding one
# later can't silently pull test/lint tooling into the runtime image.
# --no-emit-project excludes gepa itself, which is installed --no-deps after,
# so nothing gets re-resolved.
# uv pinned, and installed from PyPI rather than pulled from a second
# registry, so the image has exactly one dependency source.
RUN pip install --no-cache-dir "uv==0.11.1" \
 && uv export --locked --extra rpc --no-default-groups --no-dev \
        --no-emit-project -o /tmp/requirements.txt \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip install --no-cache-dir --no-deps . \
 && pip uninstall -y uv \
 && rm -f /tmp/requirements.txt

RUN useradd --no-create-home --shell /bin/false appuser && chown appuser /app
USER appuser

EXPOSE 50051

# A container must bind 0.0.0.0 for `docker run -p` to reach it at all, so
# the loopback default in server.py cannot protect a container. Instead the
# server refuses to start on a non-loopback address unless one of
# --auth-token / --tls-cert / --allow-insecure-remote is supplied, which
# makes the unauthenticated case a deliberate choice rather than what you get
# by following the quickstart. Supply the credential at `docker run` time:
#
#   docker run -p 50051:50051 -e OPENAI_API_KEY=... #     gepa-rpc:latest --auth-token "$GEPA_RPC_TOKEN"
#
# or, on a genuinely trusted network, --allow-insecure-remote.
ENTRYPOINT ["gepa-rpc", "--port", "50051", "--host", "0.0.0.0"]
