"""Generate the Python gRPC bindings from the shared contract in ../proto.

    uv run python scripts/generate_grpc.py            # write grpc_server/generated/
    uv run python scripts/generate_grpc.py --check    # exit 1 if they are stale

The .proto is copied under grpc_server/generated/ in a temporary tree before
protoc runs, so the generated modules import each other as
`grpc_server.generated...` rather than as a top-level `atlas` package, which
would collide with the Java project directory ../atlas on sys.path. The wire
names (package atlas.v1, service AtlasService) are untouched.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
PROTO = APP.parent / "proto" / "atlas" / "v1" / "atlas_service.proto"
PACKAGE = Path("grpc_server") / "generated"
OUTPUTS = ("atlas_service_pb2.py", "atlas_service_pb2.pyi", "atlas_service_pb2_grpc.py")


def generate(into: Path) -> None:
    from grpc_tools import protoc

    with tempfile.TemporaryDirectory() as include:
        target = Path(include) / PACKAGE / "atlas_service.proto"
        target.parent.mkdir(parents=True)
        shutil.copyfile(PROTO, target)
        into.mkdir(parents=True, exist_ok=True)
        code = protoc.main([
            "grpc_tools.protoc",
            f"-I{include}",
            f"--python_out={into}",
            f"--pyi_out={into}",
            f"--grpc_python_out={into}",
            str(target),
        ])
    if code != 0:
        raise SystemExit(f"protoc failed with exit code {code}")


def stale(directory: Path = APP) -> list:
    """Generated files that differ from what the current .proto produces."""
    with tempfile.TemporaryDirectory() as scratch:
        generate(Path(scratch))
        return [
            name for name in OUTPUTS
            if not (directory / PACKAGE / name).exists()
            or (directory / PACKAGE / name).read_bytes() != (Path(scratch) / PACKAGE / name).read_bytes()
        ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the checked-in bindings are stale")
    args = parser.parse_args(argv)
    if args.check:
        changed = stale()
        if changed:
            print("Stale gRPC bindings: " + ", ".join(changed) + "; run scripts/generate_grpc.py")
            return 1
        print("gRPC bindings are up to date")
        return 0
    generate(APP)
    init = APP / PACKAGE / "__init__.py"
    if not init.exists():
        init.write_text('"""Generated from ../proto by scripts/generate_grpc.py; do not edit."""\n')
    print(f"Generated {', '.join(OUTPUTS)} in {APP / PACKAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
