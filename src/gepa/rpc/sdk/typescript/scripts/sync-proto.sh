#!/usr/bin/env bash
# Copies the canonical proto from src/gepa/rpc/proto into the SDK package so
# it ships as part of the published npm package.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RPC_ROOT="$(cd "$HERE/../../.." && pwd)"
SRC="$RPC_ROOT/proto/gepa.proto"
DST="$HERE/../proto/gepa.proto"
if [[ ! -f "$SRC" ]]; then
  echo "Error: $SRC not found." >&2
  exit 1
fi
mkdir -p "$(dirname "$DST")"
cp "$SRC" "$DST"
echo "synced: $SRC -> $DST"
