"""Pinned Liger source and packet boundaries.

These helpers keep source identity and packet staging separate from capture and
runtime witnesses.
"""

from __future__ import annotations

from pathlib import Path

from .canonical import sha256_bytes, write_canonical_json
from .errors import TargetError
from .source_files import (
    _read_regular_file,
    _read_verified_tree_file,
)

LIGER_REPOSITORY = "https://github.com/linkedin/Liger-Kernel"
LIGER_REVISION = "e5164336825e34d0161d762c294e25d9f35649cc"
LIGER_SOURCE_PATH = "src/liger_kernel/ops/rms_norm.py"
LIGER_SOURCE_SHA256 = "869f58a0e85a18bcb71b9e1827565b4ff43c70231dd2522287ff3c3a6550655b"
LIGER_UTILS_PATH = "src/liger_kernel/ops/utils.py"
LIGER_UTILS_SHA256 = "163eee954cbfa89e2620eb496d03b9f67f26d36244fa34b420d78d8cf6ea57fb"
LIGER_ROOT_UTILS_PATH = "src/liger_kernel/utils.py"
LIGER_ROOT_UTILS_SHA256 = "cb6b94b41e1d43650b5f3c3b71f963f2080a0f36c1803025800af306da1bb2f6"
LIGER_NOTICE_PATH = "NOTICE"
LIGER_NOTICE_SHA256 = "9e3c27a0f64b87d00df12250cf1bc218b1e2fbc5fffc0bd64737ba8e8357218f"
LIGER_LICENSE_SHA256 = "3a1ccb0c7274b68e1af2ca1d54b10b662085ca56753400182ecf87ae33f2d1a8"
LIGER_TRITON_LICENSE_PATH = "licenses/LICENSE-MIT-triton"
LIGER_TRITON_LICENSE_SHA256 = "c0be96f08c2c2a2c87b735ee6e0f1309785166ea3c4bcdc43dda977df62da968"
UNSLOTH_REVISION = "fd753fed99ed5f10ef8a9b7139588d9de9ddecfb"
UNSLOTH_LICENSE_SHA256 = "29f410b6f9f7aadbd2dfdce2578afd9b4b05be9e9a52c513c97fd1dd554f3eb2"


def _validate_liger_tree(root: Path) -> None:
    for relative, expected, _role in _selected_liger_files():
        _read_verified_tree_file(root, relative, expected, "Liger")


def _selected_liger_files():
    return (
        (LIGER_SOURCE_PATH, LIGER_SOURCE_SHA256, "source"),
        (LIGER_UTILS_PATH, LIGER_UTILS_SHA256, "source"),
        (LIGER_ROOT_UTILS_PATH, LIGER_ROOT_UTILS_SHA256, "source"),
        (LIGER_NOTICE_PATH, LIGER_NOTICE_SHA256, "notice"),
        ("LICENSE", LIGER_LICENSE_SHA256, "license"),
        (LIGER_TRITON_LICENSE_PATH, LIGER_TRITON_LICENSE_SHA256, "license"),
    )


def _stage_external_sources(
    root: Path,
    output_dir: Path,
):
    _validate_liger_tree(root)
    external = output_dir / "external" / "liger"
    tree = external / "tree"
    for relative, expected, _role in _selected_liger_files():
        destination = tree / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_read_verified_tree_file(root, relative, expected, "Liger"))
    bundled_license = Path(__file__).resolve().parent / "licenses" / "UNSLOTH-APACHE-2.0.txt"
    license_bytes = _read_regular_file(
        bundled_license, "bundled Unsloth Apache-2.0 license", "Liger"
    )
    actual = sha256_bytes(license_bytes)
    if actual != UNSLOTH_LICENSE_SHA256:
        raise TargetError(f"bundled Unsloth Apache-2.0 license hash mismatch: {actual}")
    (external / "UNSLOTH-APACHE-2.0.txt").write_bytes(license_bytes)
    notice = {
        "repository": LIGER_REPOSITORY,
        "revision": LIGER_REVISION,
        "source_header_contains_upstream_attribution": True,
        "upstream": f"https://github.com/unslothai/unsloth/tree/{UNSLOTH_REVISION}",
        "license_sha256": UNSLOTH_LICENSE_SHA256,
    }
    write_canonical_json(external / "NOTICE.json", notice)


def _write_reproducer(output_dir: Path):
    script = """import sys
sys.dont_write_bytecode = True

import json
from pathlib import Path

from daggertrace.targets import run_liger_reproducer

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "external" / "liger" / "tree"
    result = run_liger_reproducer(target)
    print(json.dumps(result, sort_keys=True))
"""
    (output_dir / "reproducer.py").write_text(script, encoding="utf-8")
