#!/usr/bin/env bash
set -euo pipefail
PROTO_DIR="./src/gepa/rpc/proto"
PROTO_FILE="${PROTO_DIR}/gepa.proto"
OUT_DIR="./src/gepa/rpc/generated"
mkdir -p "${OUT_DIR}"
if [[ ! -f "${PROTO_FILE}" ]]; then
  echo "Error: ${PROTO_FILE} not found. Run this script from the repo root." >&2
  exit 1
fi
python -m grpc_tools.protoc \
  -I"${PROTO_DIR}" \
  --python_out="${OUT_DIR}" \
  --pyi_out="${OUT_DIR}" \
  --grpc_python_out="${OUT_DIR}" \
  "${PROTO_FILE}"
if [[ ! -f "${OUT_DIR}/gepa_pb2_grpc.py" ]]; then
  echo "Error: expected ${OUT_DIR}/gepa_pb2_grpc.py was not generated." >&2
  exit 1
fi
sed -i.bak 's/^import gepa_pb2 as gepa__pb2$/from gepa.rpc.generated import gepa_pb2 as gepa__pb2/' "${OUT_DIR}/gepa_pb2_grpc.py"
rm -f "${OUT_DIR}/gepa_pb2_grpc.py.bak"
# grpc stubs don't expose grpc.experimental; suppress false-positive type errors in this generated file.
python3 -c "
import sys
path = '${OUT_DIR}/gepa_pb2_grpc.py'
with open(path) as f:
    content = f.read()
if '# type: ignore' not in content[:100]:
    content = '# type: ignore\n' + content
with open(path, 'w') as f:
    f.write(content)
"
echo "Proto compilation succeeded. Generated files:"
echo "  - ${OUT_DIR}/gepa_pb2.py"
echo "  - ${OUT_DIR}/gepa_pb2.pyi"
echo "  - ${OUT_DIR}/gepa_pb2_grpc.py"
echo "Note: sdk/typescript and sdk/rust live in the gepa-polyglot repo -- sync ${PROTO_FILE} there manually if it changed."
