from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from daggertrace import cli_audit, cli_replay
from daggertrace.errors import DaggerTraceError
from daggertrace.replay import ReplayError


def test_replay_root_bounds_all_immediate_entries(tmp_path):
    for index in range(cli_replay._MAX_REPLAY_ROOT_ENTRIES + 1):
        (tmp_path / f"empty-{index}").write_text("", encoding="utf-8")

    with pytest.raises(ReplayError, match="root exceeds .* entries"):
        cli_replay._packet_paths(tmp_path)


def test_public_replay_rejects_an_unknown_target_repository(tmp_path):
    from daggertrace import targets
    from daggertrace.errors import TargetError

    with pytest.raises(TargetError, match="unsupported certificate target repository"):
        targets._replay_target(
            {"target": {"repository": "https://example.invalid/unknown"}},
            tmp_path / "certificate.json",
            {"valid": True},
        )


def test_multi_target_promotion_replaces_the_root_once(monkeypatch, tmp_path):
    from argparse import Namespace

    output = tmp_path / "final"
    target_root = tmp_path / "targets"
    for target in ("liger-rmsnorm", "openpipe-art-local-stats"):
        (target_root / target).mkdir(parents=True)
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    observed_roots = []

    def build(_args, target, staging, packet_output, target_root, _validation):
        observed_roots.append((target, target_root))
        (staging / "certificate.json").write_text(target, encoding="utf-8")
        return {
            "status": "VERIFIED SAFE_IN_FRAGMENT",
            "target": {"name": target},
            "analysis": {"outcome": "SAFE_IN_FRAGMENT", "reasons": []},
            "artifacts": {"certificate": str(packet_output / "certificate.json"), "reproducer": ""},
        }

    monkeypatch.setattr(cli_audit, "_build_audit_packet", build)
    args = Namespace(warmups=1, samples=3, json=False)

    result = cli_audit._audit_many(
        args,
        ("liger-rmsnorm", "openpipe-art-local-stats"),
        output,
        target_root,
        {"mutation_cases": 0},
    )

    assert result["directory"] == str(output)
    assert observed_roots == [
        ("liger-rmsnorm", target_root / "liger-rmsnorm"),
        ("openpipe-art-local-stats", target_root / "openpipe-art-local-stats"),
    ]
    assert not (output / "stale.txt").exists()
    assert (output / "liger-rmsnorm" / "certificate.json").read_text() == "liger-rmsnorm"
    assert (
        output / "openpipe-art-local-stats" / "certificate.json"
    ).read_text() == "openpipe-art-local-stats"
    assert not list(tmp_path.glob(".final.staging-*"))


def test_single_target_audit_migrates_a_legacy_direct_packet_atomically(
    monkeypatch, tmp_path
):
    from argparse import Namespace
    from daggertrace import validation

    output = tmp_path / "final"
    output.mkdir()
    (output / "certificate.json").write_text("legacy", encoding="utf-8")
    (output / "stale.txt").write_text("stale", encoding="utf-8")

    monkeypatch.setattr(validation, "run_semantic_validation", lambda: {})

    def build(_args, target, staging, packet_output, _target_root, _validation):
        (staging / "certificate.json").write_text(target, encoding="utf-8")
        return {
            "status": "VERIFIED SAFE_IN_FRAGMENT",
            "checker": {},
            "target_id": target,
            "target": {"repository": target, "revision": "r"},
            "analysis": {"outcome": "SAFE_IN_FRAGMENT", "reasons": []},
            "artifacts": {"certificate": str(packet_output / "certificate.json")},
        }

    monkeypatch.setattr(cli_audit, "_build_audit_packet", build)
    monkeypatch.setattr(cli_audit, "_audit_text", lambda _summary: "")
    monkeypatch.setattr(cli_audit, "_emit", lambda *_args: None)
    args = Namespace(
        output=output,
        target="openpipe-art-local-stats",
        target_root=None,
        warmups=1,
        samples=3,
        json=False,
    )

    assert cli_audit._audit(args) == 0
    assert (output / "openpipe-art-local-stats" / "certificate.json").read_text() == (
        "openpipe-art-local-stats"
    )
    assert not (output / "certificate.json").exists()
    assert not (output / "stale.txt").exists()


def test_single_target_audit_replaces_only_selected_child_in_a_valid_root(
    monkeypatch, tmp_path
):
    from argparse import Namespace
    from daggertrace import validation

    output = tmp_path / "final"
    selected = output / "liger-rmsnorm"
    sibling = output / "openpipe-art-local-stats"
    selected.mkdir(parents=True)
    sibling.mkdir(parents=True)
    (selected / "certificate.json").write_bytes(b"old-selected")
    sibling_bytes = b"old-sibling"
    (sibling / "certificate.json").write_bytes(sibling_bytes)
    monkeypatch.setattr(validation, "run_semantic_validation", lambda: {})

    def build(_args, target, staging, packet_output, _target_root, _validation):
        assert staging.parent == output.parent
        cache = output.parent / ".target-cache"
        cache.mkdir(exist_ok=True)
        (cache / "sentinel").write_bytes(b"cached")
        (staging / "certificate.json").write_bytes(b"new-selected")
        return {
            "status": "VERIFIED SAFE_IN_FRAGMENT",
            "checker": {},
            "target_id": target,
            "target": {"repository": target, "revision": "r"},
            "analysis": {"outcome": "SAFE_IN_FRAGMENT", "reasons": []},
            "artifacts": {"certificate": str(packet_output / "certificate.json")},
        }

    monkeypatch.setattr(cli_audit, "_build_audit_packet", build)
    monkeypatch.setattr(cli_audit, "_audit_text", lambda _summary: "")
    monkeypatch.setattr(cli_audit, "_emit", lambda *_args: None)
    args = Namespace(
        output=output,
        target="liger-rmsnorm",
        target_root=None,
        warmups=1,
        samples=3,
        json=False,
    )

    assert cli_audit._audit(args) == 0
    assert (selected / "certificate.json").read_bytes() == b"new-selected"
    assert (sibling / "certificate.json").read_bytes() == sibling_bytes
    assert (output.parent / ".target-cache" / "sentinel").read_bytes() == b"cached"
    assert not (output / ".target-cache").exists()


def test_multi_target_audit_keeps_the_external_target_cache(monkeypatch, tmp_path):
    from argparse import Namespace

    output = tmp_path / "final"
    cache = output.parent / ".target-cache"

    def build(_args, target, staging, packet_output, _target_root, _validation):
        cache.mkdir(exist_ok=True)
        (cache / "sentinel").write_text("cached", encoding="utf-8")
        (staging / "certificate.json").write_text(target, encoding="utf-8")
        return {
            "status": "VERIFIED SAFE_IN_FRAGMENT",
            "target": {"repository": target},
            "analysis": {"outcome": "SAFE_IN_FRAGMENT", "reasons": []},
            "artifacts": {"certificate": str(packet_output / "certificate.json")},
        }

    monkeypatch.setattr(cli_audit, "_build_audit_packet", build)
    args = Namespace(warmups=1, samples=3)
    cli_audit._audit_many(
        args,
        ("liger-rmsnorm", "openpipe-art-local-stats"),
        output,
        None,
        {"mutation_cases": 0},
    )

    assert (cache / "sentinel").read_text(encoding="utf-8") == "cached"


def test_multi_target_target_root_requires_explicit_child_layout(tmp_path):
    with pytest.raises(DaggerTraceError, match="must contain a pinned checkout"):
        cli_audit._target_root_for(tmp_path, "liger-rmsnorm", multiple=True)


def test_single_target_audit_uses_a_nested_packet_path(monkeypatch, tmp_path):
    from argparse import Namespace

    observed = []

    def audit_one(_args, target, output, target_root, semantic_validation):
        observed.append((target, output, target_root, semantic_validation))
        return {"status": "VERIFIED"}

    monkeypatch.setattr(cli_audit, "_audit_one", audit_one)
    monkeypatch.setattr(cli_audit, "_emit", lambda *_args: None)
    monkeypatch.setattr(cli_audit, "_audit_text", lambda _summary: "")
    args = Namespace(
        output=tmp_path / "final",
        target="openpipe-art-local-stats",
        target_root=None,
        warmups=1,
        samples=3,
        json=False,
    )

    assert cli_audit._audit(args) == 0
    assert observed == [
        (
            "openpipe-art-local-stats",
            (tmp_path / "final" / "openpipe-art-local-stats").resolve(),
            None,
            None,
        )
    ]


def test_multi_target_failure_keeps_the_previous_root(monkeypatch, tmp_path):
    from argparse import Namespace

    output = tmp_path / "final"
    output.mkdir()
    (output / "certificate.json").write_text("old", encoding="utf-8")

    def build(_args, target, staging, _packet_output, _target_root, _validation):
        (staging / "certificate.json").write_text(target, encoding="utf-8")
        if target == "openpipe-art-local-stats":
            raise DaggerTraceError("second target failed")
        return {}

    monkeypatch.setattr(cli_audit, "_build_audit_packet", build)
    args = Namespace(warmups=1, samples=3, json=False)

    with pytest.raises(DaggerTraceError, match="second target failed"):
        cli_audit._audit_many(
            args,
            ("liger-rmsnorm", "openpipe-art-local-stats"),
            output,
            None,
            {"mutation_cases": 0},
        )

    assert (output / "certificate.json").read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".final.staging-*"))


def test_replay_command_does_not_import_producer_modules():
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    script = """
import sys
import daggertrace.cli as cli
from daggertrace import cli_replay

cli_replay.replay_certificate = lambda _path: {
    "valid": True,
    "certificate_digest": "d",
    "outcome": "VIOLATION",
    "reasons": ["BORROWED_COTANGENT_WRITE"],
    "borrowed_writes": 1,
    "contextual_collapse": True,
}
assert cli.main(["replay", "unused.json", "--json"]) == 0
forbidden = {
    "daggertrace.certificate",
    "daggertrace.targets",
    "daggertrace.validation",
}
loaded = sorted(forbidden.intersection(sys.modules))
assert not loaded, loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
