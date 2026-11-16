#!/usr/bin/env bash
# Copies the canonical proto from src/gepa/rpc/proto into the SDK crate so it
# ships as part of the published crates.io package. Mirrors
# sdk/typescript/scripts/sync-proto.sh -- Rust commits no generated code
# (build.rs compiles proto/gepa.proto at build time via tonic_build), but the
# proto *source* copy still has to stay in sync the same way.
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
