"""Behavior tests for the import-boundary validation script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "check_architecture.py"
_SPEC = importlib.util.spec_from_file_location("check_architecture", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_CHECKER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECKER)


def _package(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "daggertrace"
    root.mkdir()
    for name, contents in files.items():
        (root / name).write_text(contents, encoding="utf-8")
    return root


def test_current_package_satisfies_import_boundaries():
    assert _CHECKER.check(_SCRIPT.parents[1] / "src" / "daggertrace") == ()


def test_rejects_top_level_internal_import_cycle(tmp_path: Path):
    root = _package(
        tmp_path,
        {
            "replay.py": "from . import replay_core\n",
            "replay_core.py": "from . import replay\n",
        },
    )

    assert _CHECKER.check(root) == (
        "top-level internal import cycle: daggertrace.replay, daggertrace.replay_core",
    )


def test_allows_function_local_facade_reference_without_a_top_level_cycle(tmp_path: Path):
    root = _package(
        tmp_path,
        {
            "art_target.py": "from . import art_capture\n",
            "art_capture.py": "def capture():\n    from . import art_target\n    return art_target\n",
        },
    )

    assert _CHECKER.check(root) == ()


def test_deferred_facades_preserve_runtime_constants(monkeypatch):
    from daggertrace import art_capture, art_target

    assert art_capture._target_module() is art_target

    monkeypatch.setattr(art_target, "ART_SHAPE", (1, 1024))
    assert art_capture._expected_art_grid("_stats_stage1_kernel") == [{"tuple": [1, 1, 1]}]


def test_rejects_replay_closure_that_reaches_runtime_capture(tmp_path: Path):
    root = _package(
        tmp_path,
        {
            "replay.py": "from . import replay_core\n",
            "replay_core.py": "from . import capture\n",
            "capture.py": "",
        },
    )

    assert _CHECKER.check(root) == (
        "dependency-light replay reaches daggertrace.replay -> daggertrace.replay_core -> daggertrace.capture",
    )


def test_rejects_replay_closure_that_reaches_a_gpu_runtime_submodule(tmp_path: Path):
    root = _package(tmp_path, {"replay.py": "import triton.runtime\n"})

    assert _CHECKER.check(root) == (
        "dependency-light replay reaches daggertrace.replay -> triton.runtime",
    )


def test_rejects_absolute_package_from_import_across_a_boundary(tmp_path: Path):
    root = _package(
        tmp_path,
        {
            "canonical.py": "from daggertrace import cli\n",
            "cli.py": "",
        },
    )

    assert _CHECKER.check(root) == (
        "core layer reaches upward dependency daggertrace.canonical -> daggertrace.cli",
    )


@pytest.mark.parametrize(
    "dependency",
    ("from . import cli", "from daggertrace import cli"),
)
def test_rejects_imports_executed_in_a_class_body(tmp_path: Path, dependency: str):
    root = _package(
        tmp_path,
        {
            "canonical.py": f"class Escape:\n    {dependency}\n",
            "cli.py": "",
        },
    )

    assert _CHECKER.check(root) == (
        "core layer reaches upward dependency daggertrace.canonical -> daggertrace.cli",
    )


@pytest.mark.parametrize(
    ("source", "dependency", "expected"),
    [
        (
            "canonical.py",
            "from . import cli\n",
            "core layer reaches upward dependency daggertrace.canonical -> daggertrace.cli",
        ),
        (
            "model_core.py",
            "from . import art_target\n",
            "core layer reaches upward dependency daggertrace.model_core -> daggertrace.art_target",
        ),
        (
            "target_support.py",
            "from . import targets\n",
            "target support reaches routing or an adapter daggertrace.target_support -> daggertrace.targets",
        ),
        (
            "target_support.py",
            "from . import art_target\n",
            "target support reaches routing or an adapter daggertrace.target_support -> daggertrace.art_target",
        ),
    ],
)
def test_rejects_upward_imports(tmp_path: Path, source: str, dependency: str, expected: str):
    module = dependency.removeprefix("from . import ").strip()
    root = _package(tmp_path, {source: dependency, f"{module}.py": ""})

    assert _CHECKER.check(root) == (expected,)
