"""Target-family boundaries for the import-direction checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "check_architecture.py"
_SPEC = importlib.util.spec_from_file_location("check_target_architecture", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_CHECKER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECKER)


def _package(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "daggertrace"
    root.mkdir()
    for name, contents in files.items():
        (root / name).write_text(contents, encoding="utf-8")
    return root


@pytest.mark.parametrize(
    "dependency",
    (
        "targets",
        "liger_capture",
        "liger_packet",
        "liger_target",
    ),
)
def test_rejects_art_adapter_imports_from_another_target_family(tmp_path: Path, dependency: str):
    root = _package(
        tmp_path,
        {
            "art_target.py": f"from . import {dependency}\n",
            f"{dependency}.py": "",
        },
    )

    assert _CHECKER.check(root) == (
        "target adapter crosses family boundary "
        f"daggertrace.art_target -> daggertrace.{dependency}",
    )


def test_allows_a_target_adapter_to_import_its_own_family(tmp_path: Path):
    root = _package(
        tmp_path,
        {
            "art_target.py": "from . import art_capture\n",
            "art_capture.py": "",
        },
    )

    assert _CHECKER.check(root) == ()


def test_rejects_an_indirect_import_from_another_target_family(tmp_path: Path):
    root = _package(
        tmp_path,
        {
            "art_target.py": "from . import adapter_bridge\n",
            "adapter_bridge.py": "from . import liger_target\n",
            "liger_target.py": "",
        },
    )

    assert _CHECKER.check(root) == (
        "target adapter crosses family boundary "
        "daggertrace.art_target -> daggertrace.adapter_bridge -> daggertrace.liger_target",
    )


def test_rejects_eager_adapter_import_from_target_router(tmp_path: Path):
    root = _package(
        tmp_path,
        {
            "targets.py": "from . import art_target\n",
            "art_target.py": "",
        },
    )

    assert _CHECKER.check(root) == (
        "target router eagerly imports adapter daggertrace.targets -> daggertrace.art_target",
    )


def test_allows_cli_to_route_a_target_adapter(tmp_path: Path):
    root = _package(
        tmp_path,
        {
            "cli_audit.py": "from . import art_target\n",
            "art_target.py": "",
        },
    )

    assert _CHECKER.check(root) == ()
