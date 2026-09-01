FROM python:3.13-slim

ARG UV_VERSION=0.8.14

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    EVALCLI_BIN=/opt/evalcli/eval_cli

RUN pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra glean

# The open-source image is useful for --fake_flow immediately. Real Cortex
# executions require a Linux evalcli bundle supplied by the internal build.
COPY deploy/glean_gepa/evalcli_bundle /opt/evalcli

ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "glean_gepa.remote_job"]
