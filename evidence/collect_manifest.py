"""Write a manifest for a generated evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from daggertrace.canonical import sha256_file, write_canonical_json  # noqa: E402


def collect(root: Path, excluded: Path | None = None) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"evidence bundle root must be a regular directory: {root}")
    if excluded is not None and excluded.is_symlink():
        raise RuntimeError(f"manifest output must not be a symlink: {excluded}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"evidence bundle contains a symlink: {path}")
        if excluded is not None and path.resolve() == excluded.resolve():
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        data = path.stat()
        files.append({"path": relative, "size": data.st_size, "sha256": sha256_file(path)})
    return {
        "schema": "daggertrace.local-evidence-manifest/1",
        "root": ".",
        "file_count": len(files),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_canonical_json(args.output, collect(args.root, args.output))
    print(json.dumps({"output": args.output.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
