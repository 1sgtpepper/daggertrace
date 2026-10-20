from __future__ import annotations

import json
from pathlib import Path

import pytest

from daggertrace import cli_replay
from daggertrace.cli import main
from daggertrace.replay import ReplayError


def test_replay_rejection_is_separate_from_cli_errors(monkeypatch, capsys):
    from daggertrace.replay import ReplayError

    monkeypatch.setattr(
        cli_replay,
        "replay_certificate",
        lambda _path: (_ for _ in ()).throw(ReplayError("certificate digest mismatch")),
    )

    assert main(["replay", "unused.json"]) == 1
    assert capsys.readouterr().err == "daggertrace: replay rejected: certificate digest mismatch\n"


def test_replay_directory_discovers_each_target_packet(monkeypatch, tmp_path, capsys):
    root = tmp_path / "final"
    for target in ("liger-rmsnorm", "openpipe-art-local-stats"):
        packet = root / target
        packet.mkdir(parents=True)
        (packet / "certificate.json").write_text(target, encoding="utf-8")

    def replay(path):
        return {
            "valid": True,
            "certificate_digest": path.read_text(encoding="utf-8"),
            "outcome": "SAFE_IN_FRAGMENT",
            "reasons": ["FRESH_DESTINATION"],
            "borrowed_writes": 0,
            "contextual_collapse": False,
        }

    monkeypatch.setattr(cli_replay, "replay_certificate", replay)

    assert main(["replay", str(root), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert {Path(packet["path"]).parent.name for packet in output["packets"]} == {
        "liger-rmsnorm",
        "openpipe-art-local-stats",
    }
    assert all(packet["result"]["valid"] is True for packet in output["packets"])


def test_replay_root_with_one_nested_packet_keeps_aggregate_shape(monkeypatch, tmp_path, capsys):
    root = tmp_path / "bundle"
    certificate = root / "nested" / "packet" / "certificate.json"
    certificate.parent.mkdir(parents=True)
    certificate.write_text("certificate", encoding="utf-8")

    monkeypatch.setattr(
        cli_replay,
        "replay_certificate",
        lambda path: {
            "valid": True,
            "certificate_digest": path.read_text(encoding="utf-8"),
            "outcome": "SAFE_IN_FRAGMENT",
            "reasons": ["FRESH_DESTINATION"],
            "borrowed_writes": 0,
            "contextual_collapse": False,
        },
    )

    assert main(["replay", str(root), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["packet_count"] == 1
    assert len(output["packets"]) == 1


def test_replay_root_summary_keeps_discovered_shape_after_root_is_renamed(
    monkeypatch, tmp_path, capsys
):
    root = tmp_path / "bundle"
    certificate = root / "nested" / "packet" / "certificate.json"
    certificate.parent.mkdir(parents=True)
    certificate.write_text("certificate", encoding="utf-8")

    def replay(_path):
        root.rename(tmp_path / "renamed")
        return {
            "valid": True,
            "certificate_digest": "certificate",
            "outcome": "SAFE_IN_FRAGMENT",
            "reasons": ["FRESH_DESTINATION"],
            "borrowed_writes": 0,
            "contextual_collapse": False,
        }

    monkeypatch.setattr(cli_replay, "replay_certificate", replay)

    assert main(["replay", str(root), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["packet_count"] == 1
    assert len(output["packets"]) == 1


def test_replay_root_rejects_an_incomplete_packet_like_child(tmp_path):
    root = tmp_path / "bundle"
    valid = root / "valid"
    incomplete = root / "incomplete"
    valid.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    (valid / "certificate.json").write_text("certificate", encoding="utf-8")
    (incomplete / "candidate.capture.json").write_text("capture", encoding="utf-8")

    with pytest.raises(ReplayError, match="incomplete packet"):
        cli_replay._packet_paths(root)


def test_replay_directory_dispatches_certificate_and_source_evidence(monkeypatch, tmp_path):
    root = tmp_path / "final"
    certificate = root / "liger-rmsnorm" / "certificate.json"
    evidence = root / "triton-layernorm" / "source-evidence.json"
    certificate.parent.mkdir(parents=True)
    evidence.parent.mkdir(parents=True)
    certificate.write_text("certificate", encoding="utf-8")
    evidence.write_text("evidence", encoding="utf-8")
    calls = []

    def replay_certificate(path):
        calls.append(("certificate", path))
        return {"valid": True, "outcome": "VIOLATION", "reasons": []}

    def replay_source(path):
        calls.append(("source", path))
        return {
            "valid": True,
            "schema": "daggertrace.aot-source-evidence/2",
            "outcome": "UNKNOWN",
            "reasons": ["MEMORY_MODEL"],
            "source_sha256": "a" * 64,
        }

    monkeypatch.setattr(cli_replay, "replay_certificate", replay_certificate)
    monkeypatch.setattr(cli_replay, "replay_source_evidence", replay_source)

    value, _human = cli_replay._replay_summary(
        [(path, cli_replay._replay_packet(path)) for path in cli_replay._packet_paths(root)]
    )

    assert value["valid"] is True
    assert value["packets"][1]["result"]["schema"] == "daggertrace.aot-source-evidence/2"
    assert calls == [("certificate", certificate), ("source", evidence)]


def test_replay_root_rejects_more_than_the_bounded_packet_count(tmp_path):
    root = tmp_path / "packets"
    for index in range(cli_replay._MAX_REPLAY_PACKETS + 1):
        packet = root / f"packet-{index:03d}"
        packet.mkdir(parents=True)
        (packet / "certificate.json").write_text("unused", encoding="utf-8")

    with pytest.raises(ReplayError, match="exceeds 64 packets"):
        cli_replay._packet_paths(root)


def test_replay_root_rejects_a_symlinked_child(tmp_path):
    root = tmp_path / "packets"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "certificate.json").write_text("unused", encoding="utf-8")
    (root / "aliased").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ReplayError, match="symlinked child"):
        cli_replay._packet_paths(root)


def test_replay_root_rejects_ambiguous_direct_and_child_layouts(tmp_path):
    root = tmp_path / "packets"
    child = root / "child"
    child.mkdir(parents=True)
    (root / "certificate.json").write_text("unused", encoding="utf-8")
    (child / "source-evidence.json").write_text("unused", encoding="utf-8")

    with pytest.raises(ReplayError, match="mixes direct and child"):
        cli_replay._packet_paths(root)


def test_replay_root_rejects_markers_mixed_with_child_packets(tmp_path):
    root = tmp_path / "packets"
    child = root / "child"
    child.mkdir(parents=True)
    (child / "certificate.json").write_text("unused", encoding="utf-8")
    (root / "runtime.json").write_text("unused", encoding="utf-8")

    with pytest.raises(ReplayError, match="markers and child packets"):
        cli_replay._packet_paths(root)


def test_replay_root_reserves_external_closure_from_recursive_discovery(tmp_path):
    root = tmp_path / "bundle"
    packet = root / "packet"
    closure = root / "external"
    packet.mkdir(parents=True)
    closure.mkdir(parents=True)
    (packet / "certificate.json").write_text("packet", encoding="utf-8")
    (closure / "certificate.json").write_text("closure", encoding="utf-8")

    assert cli_replay._packet_paths(root) == (packet / "certificate.json",)


def test_replay_root_reserves_target_caches_at_each_discovery_level(tmp_path):
    root = tmp_path / "bundle"
    target = root / "target"
    packet = target / "packet"
    packet.mkdir(parents=True)
    (packet / "certificate.json").write_text("packet", encoding="utf-8")
    for cache in (root / ".target-cache", target / ".target-cache"):
        cache.mkdir()
        (cache / "certificate.json").write_text("cache", encoding="utf-8")

    assert cli_replay._packet_paths(root) == (packet / "certificate.json",)


@pytest.mark.parametrize("direct_packet", (False, True))
@pytest.mark.parametrize(
    ("residue_name", "residue_nonempty"),
    (
        (".final.backup-residue", True),
        (".final.backup-residue", False),
        (".final.staging-residue", True),
        (".final.staging-residue", False),
    ),
)
def test_replay_root_rejects_transaction_residue_before_traversal(
    tmp_path, residue_name, residue_nonempty, direct_packet
):
    root = tmp_path / "bundle"
    packet = root if direct_packet else root / "packet"
    packet.mkdir(parents=True)
    (packet / "certificate.json").write_text("packet", encoding="utf-8")
    residue = root / residue_name
    residue.mkdir()
    if residue_nonempty:
        (residue / "previous").mkdir()
        (residue / "previous" / "certificate.json").write_text(
            "old", encoding="utf-8"
        )

    with pytest.raises(ReplayError, match="transaction residue"):
        cli_replay._packet_paths(root)


def test_replay_root_ignores_transaction_shaped_regular_files(tmp_path):
    root = tmp_path / "bundle"
    packet = root / "packet"
    packet.mkdir(parents=True)
    (packet / "certificate.json").write_text("packet", encoding="utf-8")
    (root / ".final.backup-unrelated").write_text("keep", encoding="utf-8")

    assert cli_replay._packet_paths(root) == (packet / "certificate.json",)


def test_replay_root_rejects_cumulative_directory_budget(tmp_path):
    root = tmp_path / "packets"
    root.mkdir()
    for first in range(8):
        for second in range(8):
            for third in range(16):
                (root / f"branch-{first}" / f"nested-{second}" / f"leaf-{third}").mkdir(
                    parents=True
                )

    with pytest.raises(ReplayError, match="visited directories"):
        cli_replay._packet_paths(root)


def test_replay_cli_dispatches_art_target_closure_from_the_same_loaded_certificate(
    monkeypatch, tmp_path, capsys
):
    from daggertrace import art_target

    certificate = tmp_path / "certificate.json"
    certificate.write_text("unused", encoding="utf-8")
    loaded = {"target": {"repository": art_target.ART_REPOSITORY}}
    generic = {
        "valid": True,
        "certificate_digest": "a" * 64,
        "outcome": "SAFE_IN_FRAGMENT",
        "reasons": ["FRESH_DESTINATION"],
        "borrowed_writes": 0,
        "contextual_collapse": False,
    }
    observed = []

    def closure(raw, path, generic_result):
        observed.append((raw, path, generic_result))
        return {
            "adapter": art_target.ART_TARGET_ID,
            "generic_replay_compatible": True,
        }

    def generic_replay(path, *, certificate_validator):
        updates = certificate_validator(loaded, path, generic)
        return {**generic, **updates}

    monkeypatch.setattr(cli_replay, "_replay_certificate_with_validator", generic_replay)
    monkeypatch.setattr(art_target, "_replay_art_loaded", closure)

    assert main(["replay", str(certificate), "--json"]) == 0
    output = capsys.readouterr().out
    assert f'"adapter": "{art_target.ART_TARGET_ID}"' in output
    assert observed == [(loaded, certificate, generic)]


def test_target_closure_rejection_is_a_replay_error(monkeypatch, tmp_path, capsys):
    from daggertrace.errors import TargetError

    monkeypatch.setattr(
        cli_replay,
        "_replay_certificate_with_validator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TargetError("closure mismatch")),
    )

    assert main(["replay", str(tmp_path / "certificate.json")]) == 1
    assert capsys.readouterr().err == "daggertrace: replay rejected: closure mismatch\n"


def test_target_audit_dispatch_selects_only_the_named_adapter(monkeypatch, tmp_path):
    from daggertrace import art_target, liger_target, targets
    from daggertrace.errors import TargetError

    observed = []

    def audit_art(output_dir, **kwargs):
        observed.append((output_dir, kwargs))
        return {"target": "art"}

    monkeypatch.setattr(art_target, "audit_art", audit_art)
    monkeypatch.setattr(
        liger_target,
        "audit_liger",
        lambda *_args, **_kwargs: pytest.fail("Liger audit was selected for the ART target"),
    )

    assert targets._audit_target("openpipe-art-local-stats", tmp_path) == {"target": "art"}
    assert observed == [
        (tmp_path, {"target_root": None, "warmups": 3, "samples": 10})
    ]
    with pytest.raises(TargetError, match="unknown target"):
        targets._audit_target("openpipe-art", tmp_path)


def test_external_reproducer_requires_an_explicit_pinned_checkout(capsys):
    assert main(["reproduce", "liger-rmsnorm", "--gpu"]) == 2
    assert "require an explicit --target-root" in capsys.readouterr().err


def test_single_reproducer_human_output_names_the_target(monkeypatch, tmp_path, capsys):
    from daggertrace import targets

    monkeypatch.setattr(
        targets,
        "_run_target_reproducer",
        lambda _target, _root: {"candidate_vs_reference": {"gx_max_abs": 0.0}},
        raising=False,
    )

    assert main(
        ["reproduce", "liger-rmsnorm", "--gpu", "--target-root", str(tmp_path)]
    ) == 0
    assert "Reproducer: liger-rmsnorm" in capsys.readouterr().out


def test_reproduce_all_reports_exact_runtime_target_set(monkeypatch, tmp_path, capsys):
    from daggertrace import targets

    target_root = tmp_path / "targets"
    for target in ("liger-rmsnorm", "openpipe-art-local-stats"):
        (target_root / target).mkdir(parents=True)

    def reproduce(target, _target_root):
        return {"target_id": target, "status": "ok"}

    monkeypatch.setattr(targets, "_run_target_reproducer", reproduce, raising=False)

    assert main(["reproduce", "all", "--gpu", "--target-root", str(target_root), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "VERIFIED"
    assert set(output["targets"]) == {
        "liger-rmsnorm",
        "openpipe-art-local-stats",
    }
    assert "skipped" not in output
    assert output["targets"]["liger-rmsnorm"]["target_id"] == "liger-rmsnorm"


def test_reproduce_all_preflights_every_target_root_before_running_a_witness(
    monkeypatch, tmp_path, capsys
):
    from daggertrace import targets

    target_root = tmp_path / "targets"
    (target_root / "liger-rmsnorm").mkdir(parents=True)
    invoked = []
    monkeypatch.setattr(
        targets,
        "_run_target_reproducer",
        lambda target, root: invoked.append((target, root)) or {"target": target},
        raising=False,
    )

    assert main(["reproduce", "all", "--gpu", "--target-root", str(target_root)]) == 2
    assert "openpipe-art-local-stats" in capsys.readouterr().err
    assert invoked == []
