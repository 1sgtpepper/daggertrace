from __future__ import annotations

import json
from pathlib import Path

import pytest

from daggertrace import cli_audit, cli_replay
from daggertrace.cli import main
from daggertrace.errors import DaggerTraceError


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    (
        ("warmups", 0, "warmups"),
        ("warmups", 11, "warmups"),
        ("samples", 2, "samples"),
        ("samples", 51, "samples"),
    ),
)
def test_audit_rejects_unbounded_arguments_before_staging(tmp_path, argument, value, message):
    from argparse import Namespace

    args = Namespace(
        output=tmp_path / "final",
        target="liger-rmsnorm",
        target_root=None,
        warmups=value if argument == "warmups" else 3,
        samples=value if argument == "samples" else 3,
        json=False,
    )

    with pytest.raises(DaggerTraceError, match=message):
        cli_audit._audit(args)
    assert not (tmp_path / "final").exists()


def test_cli_does_not_normalize_internal_assertions(monkeypatch):
    from daggertrace import validation

    failure = AssertionError("invariant failed")
    monkeypatch.setattr(
        validation,
        "run_semantic_validation",
        lambda: (_ for _ in ()).throw(failure),
    )
    with pytest.raises(AssertionError) as raised:
        main(["validate"])
    assert raised.value is failure


def test_atomic_promotion_replaces_a_nonempty_packet(tmp_path):
    output = tmp_path / "final"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    (output / "result.json").write_text("stale", encoding="utf-8")
    staging = tmp_path / ".final.staging-test"
    staging.mkdir()
    (staging / "certificate.json").write_text("new", encoding="utf-8")
    unrelated = tmp_path / ".final.backup-unrelated"
    unrelated.write_text("keep", encoding="utf-8")

    cli_audit._promote_directory(staging, output)

    assert sorted(path.name for path in output.iterdir()) == ["certificate.json"]
    assert (output / "certificate.json").read_text(encoding="utf-8") == "new"
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in tmp_path.glob(".final.backup-*")) == [
        unrelated.name
    ]


@pytest.mark.parametrize(
    ("output_exists", "residue_name", "residue_nonempty"),
    (
        (True, ".final.backup-existing", True),
        (False, ".final.backup-existing", False),
        (True, ".final.staging-existing", True),
        (False, ".final.staging-existing", False),
    ),
)
def test_atomic_promotion_rejects_existing_transaction_residue_before_mutation(
    tmp_path, output_exists, residue_name, residue_nonempty
):
    output = tmp_path / "final"
    if output_exists:
        output.mkdir()
        (output / "certificate.json").write_text("old", encoding="utf-8")
    staging = tmp_path / ".final.staging-current"
    staging.mkdir()
    (staging / "certificate.json").write_text("new", encoding="utf-8")
    residue = tmp_path / residue_name
    residue.mkdir()
    if residue_nonempty:
        (residue / "marker").write_text("residue", encoding="utf-8")

    with pytest.raises(DaggerTraceError, match="recoverable transaction residue"):
        cli_audit._promote_directory(staging, output)

    assert output.exists() is output_exists
    if output_exists:
        assert (output / "certificate.json").read_text(encoding="utf-8") == "old"
    assert sorted(
        path.name for path in tmp_path.iterdir() if path.name.startswith(".final.")
    ) == sorted((staging.name, residue.name))
    if residue_nonempty:
        assert (residue / "marker").read_text(encoding="utf-8") == "residue"
    else:
        assert list(residue.iterdir()) == []


def test_atomic_promotion_matches_literal_output_name_for_transaction_residue(tmp_path):
    output = tmp_path / "final[1]"
    output.mkdir()
    (output / "certificate.json").write_text("old", encoding="utf-8")
    staging = tmp_path / ".final[1].staging-current"
    staging.mkdir()
    (staging / "certificate.json").write_text("new", encoding="utf-8")
    residue = tmp_path / ".final[1].backup-existing"
    residue.mkdir()
    unrelated = tmp_path / ".finalX1.backup-existing"
    unrelated.mkdir()
    (unrelated / "marker").write_text("keep", encoding="utf-8")

    with pytest.raises(DaggerTraceError, match="recoverable transaction residue"):
        cli_audit._promote_directory(staging, output)

    assert (output / "certificate.json").read_text(encoding="utf-8") == "old"
    assert residue.is_dir()
    assert (unrelated / "marker").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("route", ("one", "single_root", "many"))
def test_audit_refuses_stale_packet_staging_before_allocating_a_new_packet(
    monkeypatch, tmp_path, route
):
    from argparse import Namespace

    target = "liger-rmsnorm"
    output_root = tmp_path / "final"
    packet_prefix = f".{output_root.name}.{target}.staging-"
    stale = tmp_path / f"{packet_prefix}stale"
    stale.mkdir()
    (stale / "marker").write_text("keep", encoding="utf-8")
    packet_allocations = []

    monkeypatch.setattr(
        cli_audit,
        "_build_audit_packet",
        lambda *_args: pytest.fail("stale packet staging reached packet build"),
    )
    real_mkdtemp = cli_audit.tempfile.mkdtemp

    def record_mkdtemp(*args, **kwargs):
        if kwargs.get("prefix") == packet_prefix:
            packet_allocations.append(kwargs["prefix"])
        return real_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(cli_audit.tempfile, "mkdtemp", record_mkdtemp)
    args = Namespace(warmups=1, samples=3)

    with pytest.raises(DaggerTraceError, match="recoverable transaction residue"):
        if route == "one":
            cli_audit._audit_one(args, target, output_root / target, None, {})
        elif route == "single_root":
            cli_audit._audit_single_root(args, target, output_root, None, {})
        else:
            cli_audit._audit_many(
                args,
                (target, "openpipe-art-local-stats"),
                output_root,
                None,
                {},
            )

    assert packet_allocations == []
    assert (stale / "marker").read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".final.staging-*"))
    assert not (output_root / target).exists()



def test_packet_staging_preserves_a_matching_regular_file(tmp_path):
    output_root = tmp_path / "final"
    prefix = ".final.liger-rmsnorm.staging-"
    unrelated = tmp_path / f"{prefix}unrelated"
    unrelated.write_text("keep", encoding="utf-8")

    staging = cli_audit._fresh_packet_staging(output_root, "liger-rmsnorm")

    assert staging.parent == tmp_path
    assert staging.name.startswith(prefix)
    assert staging.is_dir()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    cli_audit._discard_staging_directory(staging)


def test_atomic_promotion_reports_backup_cleanup_residue_after_install(monkeypatch, tmp_path):
    output = tmp_path / "final"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".final.staging-test"
    staging.mkdir()
    (staging / "certificate.json").write_text("new", encoding="utf-8")
    cleanup_error = OSError("backup cleanup failed")
    residues = []

    def fail_backup_cleanup(path):
        assert path.name == "previous"
        residues.append(path.parent)
        raise cleanup_error

    monkeypatch.setattr(cli_audit, "_delete_exact_path", fail_backup_cleanup)
    with pytest.raises(DaggerTraceError) as raised:
        cli_audit._promote_directory(staging, output)

    assert len(residues) == 1
    backup_container = residues[0]
    assert str(raised.value) == (
        "audit artifact promotion succeeded; cleanup failed; "
        f"recoverable residue: {backup_container}: {cleanup_error}"
    )
    assert raised.value.__cause__ is cleanup_error
    assert (output / "certificate.json").read_text(encoding="utf-8") == "new"
    assert (backup_container / "previous" / "old.txt").read_text(encoding="utf-8") == "old"


def test_atomic_promotion_reports_backup_container_residue_after_install(
    monkeypatch, tmp_path
):
    output = tmp_path / "final"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".final.staging-test"
    staging.mkdir()
    (staging / "certificate.json").write_text("new", encoding="utf-8")
    cleanup_error = OSError("backup container cleanup failed")
    backup_containers = []
    real_delete = cli_audit._delete_exact_path
    real_rmdir = Path.rmdir

    def record_backup_cleanup(path):
        backup_containers.append(path.parent)
        return real_delete(path)

    def fail_backup_container_cleanup(path):
        if backup_containers and path == backup_containers[0]:
            raise cleanup_error
        return real_rmdir(path)

    monkeypatch.setattr(cli_audit, "_delete_exact_path", record_backup_cleanup)
    monkeypatch.setattr(Path, "rmdir", fail_backup_container_cleanup)
    with pytest.raises(DaggerTraceError) as raised:
        cli_audit._promote_directory(staging, output)

    assert len(backup_containers) == 1
    backup_container = backup_containers[0]
    assert str(raised.value) == (
        "audit artifact promotion succeeded; cleanup failed; "
        f"recoverable residue: {backup_container}: {cleanup_error}"
    )
    assert raised.value.__cause__ is cleanup_error
    assert (output / "certificate.json").read_text(encoding="utf-8") == "new"
    assert backup_container.is_dir()
    assert list(backup_container.iterdir()) == []


@pytest.mark.parametrize("restore_fails", (False, True))
def test_atomic_promotion_retains_prior_packet_on_failure(monkeypatch, tmp_path, restore_fails):
    output = tmp_path / "final"
    output.mkdir()
    (output / "certificate.json").write_text("old", encoding="utf-8")
    cache = tmp_path / ".target-cache"
    cache.mkdir()
    (cache / "sentinel").write_bytes(b"cached")
    staging = tmp_path / ".final.staging-test"
    staging.mkdir()
    (staging / "certificate.json").write_text("new", encoding="utf-8")
    failure = OSError("promotion failed")
    rollback_error = OSError("restoration failed")
    replace = cli_audit._replace_path
    calls = 0

    def fail_promotion(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        if restore_fails and calls == 3:
            raise rollback_error
        replace(source, destination)

    monkeypatch.setattr(cli_audit, "_replace_path", fail_promotion)
    with pytest.raises(DaggerTraceError) as raised:
        cli_audit._promote_directory(staging, output)

    assert raised.value.__cause__ is failure
    assert (cache / "sentinel").read_bytes() == b"cached"
    assert (staging / "certificate.json").read_text(encoding="utf-8") == "new"
    backups = list(tmp_path.glob(".final.backup-*"))
    if restore_fails:
        assert not output.exists()
        assert len(backups) == 1
        prior = backups[0] / "previous"
        assert (prior / "certificate.json").read_text(encoding="utf-8") == "old"
        assert str(prior) in str(raised.value)
        assert str(rollback_error) in str(raised.value)
    else:
        assert (output / "certificate.json").read_text(encoding="utf-8") == "old"
        assert not backups


def test_atomic_promotion_preserves_primary_failure_when_rollback_cleanup_fails(
    monkeypatch, tmp_path
):
    output = tmp_path / "final"
    output.mkdir()
    (output / "certificate.json").write_text("old", encoding="utf-8")
    staging = tmp_path / ".final.staging-test"
    staging.mkdir()
    (staging / "certificate.json").write_text("new", encoding="utf-8")
    promotion_error = OSError("promotion failed")
    cleanup_error = OSError("backup container cleanup failed")
    backup_containers = []
    operations = []
    replace = cli_audit._replace_path
    real_rmdir = Path.rmdir

    def fail_promotion(source, destination):
        operations.append((source, destination))
        if source == output:
            backup_containers.append(destination.parent)
            return replace(source, destination)
        if source == staging:
            raise promotion_error
        return replace(source, destination)

    def fail_backup_container_cleanup(path):
        if backup_containers and path == backup_containers[0]:
            raise cleanup_error
        return real_rmdir(path)

    monkeypatch.setattr(cli_audit, "_replace_path", fail_promotion)
    monkeypatch.setattr(Path, "rmdir", fail_backup_container_cleanup)
    with pytest.raises(DaggerTraceError) as raised:
        cli_audit._promote_directory(staging, output)

    backup_container = backup_containers[0]
    assert len(operations) == 3
    assert operations[1] == (staging, output)
    assert operations[2] == (backup_container / "previous", output)
    assert str(raised.value) == (
        "could not atomically promote the audit artifact: "
        f"{promotion_error}; cleanup failed; recoverable residue: "
        f"{backup_container}: {cleanup_error}"
    )
    assert "promotion succeeded" not in str(raised.value)
    assert raised.value.__cause__ is promotion_error
    assert (output / "certificate.json").read_text(encoding="utf-8") == "old"
    assert backup_container.is_dir()
    assert list(backup_container.iterdir()) == []


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_atomic_promotion_cleans_backup_container_when_initial_backup_move_fails(
    monkeypatch, tmp_path, cleanup_fails
):
    output = tmp_path / "final"
    output.mkdir()
    (output / "certificate.json").write_text("old", encoding="utf-8")
    staging = tmp_path / ".final.staging-test"
    staging.mkdir()
    (staging / "certificate.json").write_text("new", encoding="utf-8")
    promotion_error = OSError("backup move failed")
    cleanup_error = OSError("backup container cleanup failed")
    backup_containers = []
    operations = []
    real_rmdir = Path.rmdir

    def fail_initial_backup_move(source, destination):
        operations.append((source, destination))
        if source == output:
            backup_containers.append(destination.parent)
            raise promotion_error
        pytest.fail("initial backup failure attempted a nonexistent restore")

    def fail_backup_container_cleanup(path):
        if backup_containers and path == backup_containers[0]:
            raise cleanup_error
        return real_rmdir(path)

    monkeypatch.setattr(cli_audit, "_replace_path", fail_initial_backup_move)
    if cleanup_fails:
        monkeypatch.setattr(Path, "rmdir", fail_backup_container_cleanup)
    with pytest.raises(DaggerTraceError) as raised:
        cli_audit._promote_directory(staging, output)

    assert len(operations) == 1
    assert operations[0][0] == output
    assert raised.value.__cause__ is promotion_error
    assert (output / "certificate.json").read_text(encoding="utf-8") == "old"
    assert (staging / "certificate.json").read_text(encoding="utf-8") == "new"
    backup_container = backup_containers[0]
    if cleanup_fails:
        assert str(raised.value) == (
            f"could not replace the audit output: {promotion_error}; "
            "cleanup failed; recoverable residue: "
            f"{backup_container}: {cleanup_error}"
        )
        assert backup_container.is_dir()
        assert list(backup_container.iterdir()) == []
    else:
        assert str(raised.value) == f"could not replace the audit output: {promotion_error}"
        assert not backup_container.exists()


def test_multi_target_retry_rejects_child_backup_residue_after_single_target_failure(
    monkeypatch, tmp_path
):
    from argparse import Namespace

    target = "liger-rmsnorm"
    output_root = tmp_path / "final"
    selected = output_root / target
    selected.mkdir(parents=True)
    (selected / "marker.txt").write_text("old", encoding="utf-8")
    cleanup_error = OSError("backup cleanup failed")
    real_delete = cli_audit._delete_exact_path

    def fail_backup_cleanup(path):
        if path.name == "previous":
            raise cleanup_error
        return real_delete(path)

    def build(_args, _target, staging, _packet_output, _target_root, _validation):
        (staging / "marker.txt").write_text("new", encoding="utf-8")
        return {}

    monkeypatch.setattr(cli_audit, "_build_audit_packet", build)
    monkeypatch.setattr(cli_audit, "_delete_exact_path", fail_backup_cleanup)
    args = Namespace(warmups=1, samples=3)

    with pytest.raises(DaggerTraceError, match="promotion succeeded; cleanup failed"):
        cli_audit._audit_one(args, target, selected, None, {})

    backups = list(output_root.glob(f".{target}.backup-*"))
    assert len(backups) == 1
    backup = backups[0]
    assert (selected / "marker.txt").read_text(encoding="utf-8") == "new"
    assert (backup / "previous" / "marker.txt").read_text(encoding="utf-8") == "old"

    monkeypatch.setattr(cli_audit, "_delete_exact_path", real_delete)
    with pytest.raises(DaggerTraceError, match="recoverable transaction residue") as raised:
        cli_audit._audit_many(
            args,
            (target, "openpipe-art-local-stats"),
            output_root,
            None,
            {},
        )

    assert str(backup) in str(raised.value)
    assert (selected / "marker.txt").read_text(encoding="utf-8") == "new"
    assert (backup / "previous" / "marker.txt").read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".final.staging-*"))
    assert not list(tmp_path.glob(".final.*.staging-*"))


def test_failed_audit_preserves_prior_packet_and_removes_staging(monkeypatch, tmp_path):
    from argparse import Namespace

    from daggertrace import targets, validation

    output = tmp_path / "final"
    output.mkdir()
    (output / "certificate.json").write_text("old", encoding="utf-8")
    cache = tmp_path / ".target-cache"
    cache.mkdir()
    (cache / "sentinel").write_bytes(b"cached")

    monkeypatch.setattr(validation, "run_semantic_validation", lambda: {})

    def fail_after_partial_write(staging, **_kwargs):
        assert staging.parent == output.parent
        assert staging != output
        (staging / "partial.json").write_text("partial", encoding="utf-8")
        raise DaggerTraceError("capture failed")

    monkeypatch.setattr(
        targets,
        "_audit_target",
        lambda _target, staging, **kwargs: fail_after_partial_write(staging, **kwargs),
        raising=False,
    )
    args = Namespace(
        output=output,
        target="liger-rmsnorm",
        target_root=None,
        warmups=1,
        samples=3,
        json=False,
    )
    with pytest.raises(DaggerTraceError, match="capture failed"):
        cli_audit._audit(args)

    assert (output / "certificate.json").read_text(encoding="utf-8") == "old"
    assert (cache / "sentinel").read_bytes() == b"cached"
    assert not list(tmp_path.glob(".final.staging-*"))


@pytest.mark.parametrize("error_type", [DaggerTraceError, AssertionError])
def test_audit_preserves_primary_error_when_cleanup_fails(monkeypatch, tmp_path, error_type):
    from argparse import Namespace

    from daggertrace import validation

    primary_error = error_type("audit failed")
    cleanup_error = OSError("cleanup failed")
    cleanup_attempts = []

    def fail_audit():
        raise primary_error

    def fail_cleanup(path):
        cleanup_attempts.append(path)
        raise cleanup_error

    monkeypatch.setattr(validation, "run_semantic_validation", fail_audit)
    monkeypatch.setattr(cli_audit.shutil, "rmtree", fail_cleanup)
    args = Namespace(
        output=tmp_path / "final",
        target="liger-rmsnorm",
        target_root=None,
        warmups=1,
        samples=3,
        json=False,
    )

    with pytest.raises(error_type) as raised:
        cli_audit._audit(args)

    assert raised.value is primary_error
    assert cleanup_attempts


@pytest.mark.parametrize("killed", (34, 33))
def test_audit_seals_once_and_promotes_only_after_mutation_count_matches(
    monkeypatch, tmp_path, killed
):
    from argparse import Namespace

    from daggertrace import certificate, targets, validation, validation_mutations

    class Analysis:
        def to_dict(self):
            return {"outcome": "VIOLATION"}

    output = tmp_path / "final"
    prior_packet = output / "liger-rmsnorm"
    prior_packet.mkdir(parents=True)
    (prior_packet / "certificate.json").write_text("old", encoding="utf-8")
    run = {
        "candidate": object(),
        "control": object(),
        "analysis": Analysis(),
        "target": {},
        "runtime": {},
        "benchmark": {},
        "controls": {},
    }
    certificates = []
    replayed = []
    reproducer_writes = []

    monkeypatch.setattr(
        validation, "run_semantic_validation", lambda: {"mutation_cases": 34}
    )

    def audit_target(_target, output_dir, **_kwargs):
        (output_dir / "reproducer.py").write_bytes(b"reproducer")
        return run

    monkeypatch.setattr(targets, "_audit_target", audit_target, raising=False)

    def make_certificate(**_kwargs):
        value = {"checker": {}, "sequence": len(certificates) + 1}
        certificates.append(value)
        return value

    def write_certificate(path, value, **_kwargs):
        reproducer_writes.append(_kwargs["reproducer_bytes"])
        path.write_text(str(value["sequence"]), encoding="utf-8")

    def replay_certificate(path):
        replayed.append((path.name, path.read_text(encoding="utf-8")))
        return {"valid": True}

    def validate_mutations(value, path, **kwargs):
        assert value["sequence"] == 1
        assert path.name == "certificate.json"
        assert path.read_text(encoding="utf-8") == "1"
        assert kwargs["replay_fn"] is cli_replay._replay_mutation
        return killed

    monkeypatch.setattr(certificate, "make_certificate", make_certificate)
    monkeypatch.setattr(certificate, "write_certificate", write_certificate)
    monkeypatch.setattr(cli_replay, "replay_certificate", replay_certificate)
    monkeypatch.setattr(
        validation_mutations, "validate_certificate_mutations", validate_mutations
    )
    monkeypatch.setattr(cli_audit, "_audit_text", lambda _summary: "")
    emitted = []
    monkeypatch.setattr(cli_audit, "_emit", lambda value, *_args: emitted.append(value))

    args = Namespace(
        output=output,
        target="liger-rmsnorm",
        target_root=None,
        warmups=1,
        samples=3,
        json=False,
    )
    if killed == 34:
        assert cli_audit._audit(args) == 0
    else:
        with pytest.raises(AssertionError, match="certificate mutation total changed"):
            cli_audit._audit(args)

    assert replayed == [("certificate.json", "1")]
    assert reproducer_writes == [b"reproducer"]
    assert len(certificates) == 1
    assert not list(tmp_path.glob(".final.liger-rmsnorm.staging-*"))
    if killed == 34:
        assert (
            output / "liger-rmsnorm" / "certificate.json"
        ).read_text(encoding="utf-8") == "1"
        assert emitted[0]["validation"] == {
            "mutation_cases": 34,
            "mutations_killed": 34,
        }
    else:
        assert (prior_packet / "certificate.json").read_text(encoding="utf-8") == "old"
        assert emitted == []


def test_candidate_only_audit_reuses_target_validation_without_paired_mutations(
    monkeypatch, tmp_path
):
    from argparse import Namespace

    from daggertrace import certificate, targets, validation, validation_mutations

    class Analysis:
        def to_dict(self):
            return {"outcome": "SAFE_IN_FRAGMENT", "reasons": []}

    target = {"repository": "https://example.invalid/candidate-only", "revision": "a" * 40}
    validation_record = {
        "oracle_cases": 0,
        "oracle_passed": 0,
        "mutation_cases": 0,
        "mutations_killed": 0,
        "ablations": 0,
        "ablations_observed": 0,
    }
    run = {
        "candidate": object(),
        "control": None,
        "analysis": Analysis(),
        "target": target,
        "runtime": {},
        "benchmark": {},
        "controls": {},
    }
    replayed = []
    emitted = []

    def audit_target(_target, staging, **_kwargs):
        certificate = {
            "checker": {"owner": "target"},
            "target": target,
            "candidate": {"lane": "candidate"},
            "control": None,
            "analysis": run["analysis"].to_dict(),
            "runtime": {},
            "benchmark": {},
            "controls": {},
            "validation": validation_record,
        }
        (staging / "certificate.json").write_text(json.dumps(certificate), encoding="utf-8")
        return run

    monkeypatch.setattr(validation, "run_semantic_validation", lambda: {"mutation_cases": 34})
    monkeypatch.setattr(targets, "_audit_target", audit_target, raising=False)
    monkeypatch.setattr(
        validation_mutations,
        "validate_certificate_mutations",
        lambda *_args, **_kwargs: pytest.fail("candidate-only audit invoked paired mutations"),
    )
    monkeypatch.setattr(
        certificate,
        "make_certificate",
        lambda **_kwargs: pytest.fail("candidate-only audit rebuilt the target certificate"),
    )
    monkeypatch.setattr(
        cli_replay,
        "_replay_checked",
        lambda path: replayed.append(path) or {"valid": True},
    )
    monkeypatch.setattr(cli_audit, "_audit_text", lambda _summary: "")
    monkeypatch.setattr(cli_audit, "_emit", lambda value, *_args: emitted.append(value))

    args = Namespace(
        output=tmp_path / "final",
        target="openpipe-art-local-stats",
        target_root=None,
        warmups=1,
        samples=3,
        json=False,
    )

    assert cli_audit._audit(args) == 0
    assert len(replayed) == 1
    assert emitted[0]["validation"] == validation_record
    assert json.loads(
        (tmp_path / "final" / "openpipe-art-local-stats" / "certificate.json").read_text()
    )["control"] is None


def test_audit_raises_cleanup_error_when_audit_succeeds(monkeypatch, tmp_path):
    from argparse import Namespace

    from daggertrace import certificate, targets, validation, validation_mutations

    class Analysis:
        def to_dict(self):
            return {"outcome": "VIOLATION"}

    output = tmp_path / "final"
    run = {
        "candidate": object(),
        "control": object(),
        "analysis": Analysis(),
        "target": {},
        "runtime": {},
        "benchmark": {},
        "controls": {},
    }
    cleanup_error = OSError("cleanup failed")

    monkeypatch.setattr(validation, "run_semantic_validation", lambda: {"mutation_cases": 0})

    def audit_target(_target, output_dir, **_kwargs):
        (output_dir / "reproducer.py").write_bytes(b"reproducer")
        return run

    monkeypatch.setattr(targets, "_audit_target", audit_target, raising=False)
    monkeypatch.setattr(certificate, "make_certificate", lambda **_kwargs: {"checker": {}})
    monkeypatch.setattr(
        certificate,
        "write_certificate",
        lambda path, _value, **_kwargs: path.write_text("certificate", encoding="utf-8"),
    )
    monkeypatch.setattr(cli_replay, "replay_certificate", lambda _path: {"valid": True})
    monkeypatch.setattr(
        validation_mutations,
        "validate_certificate_mutations",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(cli_audit, "_audit_text", lambda _summary: "")
    monkeypatch.setattr(cli_audit, "_emit", lambda *_args: None)
    monkeypatch.setattr(cli_audit, "_promote_directory", lambda *_args: None)

    def fail_cleanup(_path):
        raise cleanup_error

    monkeypatch.setattr(cli_audit.shutil, "rmtree", fail_cleanup)

    args = Namespace(
        output=output,
        target="liger-rmsnorm",
        target_root=None,
        warmups=1,
        samples=3,
        json=False,
    )
    with pytest.raises(DaggerTraceError, match="could not remove audit staging directory") as raised:
        cli_audit._audit(args)

    assert raised.value.__cause__ is cleanup_error


@pytest.mark.parametrize("route", ("one", "single_root", "many"))
def test_audit_cleanup_error_is_not_swallowed_by_an_unrelated_caller_exception(
    monkeypatch, tmp_path, route
):
    from argparse import Namespace

    cleanup_error = DaggerTraceError("cleanup sentinel")
    monkeypatch.setattr(cli_audit, "_build_audit_packet", lambda *_args: {})
    monkeypatch.setattr(cli_audit, "_replace_path", lambda *_args: None)
    monkeypatch.setattr(cli_audit, "_promote_directory", lambda *_args: None)

    def fail_cleanup(_path):
        raise cleanup_error

    monkeypatch.setattr(cli_audit, "_discard_staging_directory", fail_cleanup)
    args = Namespace(warmups=1, samples=3)

    try:
        raise RuntimeError("unrelated caller failure")
    except RuntimeError:
        with pytest.raises(DaggerTraceError) as raised:
            if route == "one":
                cli_audit._audit_one(
                    args,
                    "liger-rmsnorm",
                    tmp_path / "final" / "liger-rmsnorm",
                    None,
                    {},
                )
            elif route == "single_root":
                cli_audit._audit_single_root(
                    args,
                    "liger-rmsnorm",
                    tmp_path / "final",
                    None,
                    {},
                )
            else:
                cli_audit._audit_many(
                    args,
                    ("liger-rmsnorm", "openpipe-art-local-stats"),
                    tmp_path / "final",
                    None,
                    {},
                )

    assert raised.value is cleanup_error


@pytest.mark.parametrize(
    "unsafe",
    [
        Path("/"),
        Path.cwd(),
        Path.home(),
        Path(".."),
        Path.cwd().parents[1],
        Path(Path.cwd().anchor) / "daggertrace-output",
    ],
)
def test_audit_output_rejects_broad_paths(unsafe):
    with pytest.raises(DaggerTraceError, match="unsafe audit output"):
        cli_audit._validated_output_path(unsafe)


def test_audit_output_accepts_scoped_child_and_sibling_paths(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    working_directory = workspace / "repository"
    working_directory.mkdir(parents=True)
    monkeypatch.chdir(working_directory)

    child = working_directory / "artifacts" / "final"
    sibling = workspace / "packet" / "final"

    assert cli_audit._validated_output_path(child) == child.resolve()
    assert cli_audit._validated_output_path(sibling) == sibling.resolve()


def test_audit_output_rejects_a_symlinked_directory(tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    alias = tmp_path / "output"
    alias.symlink_to(destination, target_is_directory=True)
    with pytest.raises(DaggerTraceError, match="symlinked audit output"):
        cli_audit._validated_output_path(alias)


def test_audit_output_rejects_a_symlinked_parent(tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(destination, target_is_directory=True)

    with pytest.raises(DaggerTraceError, match="symlinked audit output"):
        cli_audit._validated_output_path(alias / "packet")


@pytest.mark.parametrize("relation", ["same", "output-inside-target", "target-inside-output"])
def test_audit_rejects_output_target_overlap_before_staging(tmp_path, relation):
    from argparse import Namespace

    target = tmp_path / "target"
    output = {
        "same": target,
        "output-inside-target": target / "artifacts",
        "target-inside-output": tmp_path / "bundle",
    }[relation]
    if relation == "target-inside-output":
        target = output / "target"
    args = Namespace(
        output=output,
        target="liger-rmsnorm",
        target_root=target,
        warmups=1,
        samples=3,
        json=False,
    )
    with pytest.raises(DaggerTraceError, match="must not overlap"):
        cli_audit._audit(args)
    assert not output.exists()


def test_audit_rejects_output_overlapping_the_implicit_target_cache(tmp_path):
    from argparse import Namespace

    output = tmp_path / ".target-cache"
    args = Namespace(
        output=output,
        target="liger-rmsnorm",
        target_root=None,
        warmups=1,
        samples=3,
        json=False,
    )

    with pytest.raises(DaggerTraceError, match="implicit target cache"):
        cli_audit._audit(args)
    assert not output.exists()


def test_audit_rejects_a_symlinked_implicit_cache_overlapping_output(tmp_path):
    from argparse import Namespace

    final = tmp_path / "final"
    final.mkdir()
    (final / ".target-cache").symlink_to(final, target_is_directory=True)
    args = Namespace(
        output=final / "packet",
        target="liger-rmsnorm",
        target_root=None,
        warmups=1,
        samples=3,
        json=False,
    )

    with pytest.raises(DaggerTraceError, match="implicit target cache"):
        cli_audit._audit(args)
    assert not (final / "packet").exists()


def test_audit_accepts_a_valid_sibling_implicit_cache(monkeypatch, tmp_path):
    from argparse import Namespace

    from daggertrace import targets, validation

    final = tmp_path / "final"
    final.mkdir()
    cache = final / ".target-cache"
    cache.mkdir()
    monkeypatch.setattr(validation, "run_semantic_validation", lambda: {})
    monkeypatch.setattr(
        targets,
        "_audit_target",
        lambda _target, _staging, **_kwargs: (_ for _ in ()).throw(
            DaggerTraceError("stopped after preflight")
        ),
        raising=False,
    )
    args = Namespace(
        output=final / "packet",
        target="liger-rmsnorm",
        target_root=None,
        warmups=1,
        samples=3,
        json=False,
    )

    with pytest.raises(DaggerTraceError, match="stopped after preflight"):
        cli_audit._audit(args)
    assert cache.is_dir()


def test_audit_passes_the_resolved_target_root_to_the_gpu_owner(monkeypatch, tmp_path):
    from argparse import Namespace

    from daggertrace import targets, validation

    observed = []
    target_root = Path("~")

    monkeypatch.setattr(validation, "run_semantic_validation", lambda: {})

    def stop_before_gpu(_staging, *, target_root, **_kwargs):
        observed.append(target_root)
        raise DaggerTraceError("stop before capture")

    monkeypatch.setattr(
        targets,
        "_audit_target",
        lambda _target, staging, **kwargs: stop_before_gpu(staging, **kwargs),
        raising=False,
    )
    args = Namespace(
        output=tmp_path / "final",
        target="liger-rmsnorm",
        target_root=target_root,
        warmups=1,
        samples=3,
        json=False,
    )

    with pytest.raises(DaggerTraceError, match="stop before capture"):
        cli_audit._audit(args)
    assert observed == [Path.home().resolve()]
