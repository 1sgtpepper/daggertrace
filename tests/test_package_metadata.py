from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

import daggertrace
from daggertrace.canonical import sha256_file
from evidence.collect_manifest import collect as collect_manifest


def test_project_metadata_contract():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    source_version = (root / "src/daggertrace/version.py").read_text(encoding="utf-8")

    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    source_match = re.search(r'^__version__ = "([^"]+)"$', source_version, re.MULTILINE)
    script = re.search(r'^daggertrace = "([^"]+)"$', pyproject, re.MULTILINE)
    assert match is not None
    assert source_match is not None
    assert script is not None
    assert match.group(1) == source_match.group(1) == daggertrace.__version__ == "0.4.0"
    assert script.group(1) == "daggertrace.cli:main"
    assert re.search(r'^license = "MIT"$', pyproject, re.MULTILINE)


def test_installed_distribution_matches_source_version_and_entry_point():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    script = re.search(r'^daggertrace = "([^"]+)"$', pyproject, re.MULTILINE)
    assert version is not None
    assert script is not None

    distribution = importlib.metadata.distribution("daggertrace")
    assert distribution.version == daggertrace.__version__ == version.group(1)
    assert distribution.metadata["License-Expression"] == "MIT"
    assert any(
        entry.name == "daggertrace"
        and entry.value == script.group(1)
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    )


def test_bundled_triton_sources_stay_in_sync_with_the_test_fixtures():
    root = Path(__file__).resolve().parents[1]
    fixture_root = root / "tests/fixtures/external/triton"
    bundled_root = root / "evidence/results/bundle/triton-layernorm/external/triton/tree"
    fixture_files = {
        path.relative_to(fixture_root).as_posix()
        for path in fixture_root.rglob("*")
        if path.is_file()
    }
    bundled_files = {
        path.relative_to(bundled_root).as_posix()
        for path in bundled_root.rglob("*")
        if path.is_file()
    }

    assert fixture_files == bundled_files
    for relative in fixture_files:
        fixture = fixture_root / relative
        bundled = bundled_root / relative
        assert fixture.read_bytes() == bundled.read_bytes(), relative


def test_replay_import_does_not_load_optional_gpu_dependencies():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import daggertrace.replay; "
            "assert not {'packaging', 'torch', 'triton'} & set(sys.modules)",
        ],
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_checked_in_evidence_manifest_matches_every_sibling_file():
    root = Path(__file__).resolve().parents[1]
    bundle = root / "evidence/results/bundle"
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = collect_manifest(bundle, manifest_path)

    assert manifest["schema"] == actual["schema"]
    assert manifest["root"] == actual["root"] == "."
    assert manifest["file_count"] == actual["file_count"]
    assert manifest["files"] == actual["files"]
    assert all(
        item["sha256"] == sha256_file(bundle / item["path"])
        for item in manifest["files"]
    )


def test_checked_evidence_identity_hashes_are_fixed_and_versioned():
    root = Path(__file__).resolve().parents[1]
    bundle = root / "evidence/results/bundle"
    expected = {
        "holdout.json": "ea6d9aaae86a610ce8a0633509af560c5412d73573edc4a514cad3246b1e85a4",
        "review.json": "8cff97105227c796ff5724665c5ece831b14288561213c4115f59e908eaa9b58",
        "manifest.json": "0db6885e0ed7f839a037ace610b8d31dd3706165d5bbe83cc39501ae60b550ba",
        "liger/liger-rmsnorm/certificate.json": "ca299e2b696b26e397ff8f88f3dd0be982af03fce8b2d4a8707380d4cce8f8d3",
        "liger/liger-rmsnorm/candidate.capture.json": "29b29c7f9ab8fcf79b9ccaafbf3159612f941b5bb0f2e9ba9ad85a9427f57c90",
        "liger/liger-rmsnorm/control.capture.json": "cebc1515770e5a42cb1e29fc5dd55bda3f089149fe325f92fad75769e3419aba",
        "art/openpipe-art-local-stats/certificate.json": "d785a8c3bc8dcf978e653c6547010660cd1b26a0125c6353dfac0e25378abb57",
        "art/openpipe-art-local-stats/candidate.capture.json": "c98aa4c6b3fc3a2237ea285ed6a43f2d6dedcb897865df08188fd43bdab752f9",
        "art/openpipe-art-local-stats/control.capture.json": "38e0b9de817f645c4bec37c0d4a3e58baecccb040f5718dc069a72c7385a0bed",
    }
    assert {
        path: sha256_file(bundle / path) for path in expected
    } == expected
    assert json.loads((bundle / "holdout.json").read_text(encoding="utf-8"))["schema"] == "daggertrace.local-holdout/3"
    assert json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))["schema"] == "daggertrace.local-evidence-manifest/1"


@pytest.mark.package
def test_sdist_contains_developer_validation_tree_and_excludes_generated_packets(tmp_path):
    pytest.importorskip("build")
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    shutil.copytree(
        root,
        source,
        ignore=shutil.ignore_patterns(
            ".git",
            "build",
            "dist",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "*.egg-info",
            "__pycache__",
            "*.pyc",
        ),
    )
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=DaggerTrace tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-q",
            "-m",
            "test source archive",
        ],
        cwd=source,
        check=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    generated = source / "artifacts/final/generated/certificate.json"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_dist.py",
            "--sdist",
            "--outdir",
            str(output),
        ],
        cwd=source,
        check=True,
    )
    assert not (source / "SOURCE_REVISION").exists()
    archives = sorted(output.glob("*.tar.gz"))
    assert len(archives) == 1

    with tarfile.open(archives[0], "r:gz") as archive:
        members = archive.getmembers()
        prefix = members[0].name.split("/", 1)[0]
        revision_member = archive.extractfile(f"{prefix}/SOURCE_REVISION")
        assert revision_member is not None
        archived_revision = revision_member.read().decode("utf-8").strip()
    member_names = [member.name for member in members]
    assert archived_revision == revision
    relative_members = {
        name.removeprefix(f"{prefix}/")
        for name in member_names
        if name.startswith(f"{prefix}/")
    }
    archive_final_files = {
        member.name.removeprefix(f"{prefix}/")
        for member in members
        if member.isfile()
        and member.name.startswith(f"{prefix}/artifacts/final/")
    }
    assert not archive_final_files
    assert "(../evidence/results/bundle/)" in (
        source / "artifacts/README.md"
    ).read_text(encoding="utf-8")
    assert {
        "LICENSE",
        "CONTRIBUTING.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md",
        "Makefile",
        "SOURCE_REVISION",
        "artifacts/README.md",
        "evidence/README.md",
        "evidence/results/README.md",
        "evidence/results/bundle/manifest.json",
        "evidence/results/bundle/liger/liger-rmsnorm/certificate.json",
        "evidence/results/bundle/art/openpipe-art-local-stats/certificate.json",
        "scripts/build_dist.py",
        "scripts/validate.sh",
        "src/daggertrace/licenses/UNSLOTH-APACHE-2.0.txt",
        "tests/fixtures/external/triton/LICENSE",
        "tests/test_package_metadata.py",
    } <= relative_members
    assert all(
        "__pycache__" not in Path(name).parts
        and Path(name).suffix not in {".pyc", ".pyo", ".pyd"}
        for name in relative_members
    )
