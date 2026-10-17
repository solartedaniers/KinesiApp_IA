"""Genero el código Python de los contratos (contracts/proto) en generated/.

Uso: python scripts/generate_grpc.py
"""

import sys
from importlib.resources import files
from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parents[1]
PROTO_ROOT = ROOT / "contracts" / "proto"
OUTPUT_DIR = ROOT / "generated"


def main() -> int:
    protos = sorted(path.relative_to(PROTO_ROOT).as_posix() for path in PROTO_ROOT.rglob("*.proto"))
    if not protos:
        print(f"No .proto files under {PROTO_ROOT}: run git submodule update --init", file=sys.stderr)
        return 1
    OUTPUT_DIR.mkdir(exist_ok=True)
    # Los tipos conocidos (google/protobuf/*.proto) vienen dentro de grpcio-tools, no en el contrato
    well_known_types = files("grpc_tools") / "_proto"
    return protoc.main([
        "grpc_tools.protoc",
        f"-I{PROTO_ROOT}",
        f"-I{well_known_types}",
        f"--python_out={OUTPUT_DIR}",
        f"--pyi_out={OUTPUT_DIR}",
        f"--grpc_python_out={OUTPUT_DIR}",
        *protos,
    ])


if __name__ == "__main__":
    sys.exit(main())
