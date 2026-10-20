from __future__ import annotations

import copy
from pathlib import Path

import pytest

import daggertrace.certificate as certificate_module
from daggertrace.canonical import canonical_json_bytes
from daggertrace.certificate import make_certificate, write_certificate
from daggertrace.ownership import analyze
from daggertrace.replay import ReplayError
from daggertrace.validation_v2_fixtures import _micrograph
from daggertrace.version import CERTIFICATE_SCHEMA

from _certificate_fixtures import GENERIC_REPRODUCER, _packet
def test_v2_writer_preserves_compatible_mapping_subclasses(tmp_path):
    _, certificate = _packet(tmp_path / "seed")

    class CompatibleDict(dict):
        pass

    path = tmp_path / "v2" / "certificate.json"
    source = tmp_path / "seed" / "external" / "source" / "kernel.py"
    destination = path.parent / "external" / "source" / "kernel.py"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())
    write_certificate(path, CompatibleDict(certificate), reproducer_bytes=GENERIC_REPRODUCER)
    assert path.exists()


def test_make_certificate_snapshots_v2_caller_roots():
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    target = {"nested": {"values": [1]}}
    runtime = {"nested": {"device": "synthetic"}}
    certificate = make_certificate(
        candidate=candidate,
        control=None,
        analysis=analyze(candidate),
        target=target,
        runtime=runtime,
        benchmark={},
        controls={},
        validation={},
    )

    target["nested"]["values"].append(2)
    runtime["nested"]["device"] = "caller-mutated"
    assert certificate["target"] == {"nested": {"values": [1]}}
    assert certificate["runtime"] == {"nested": {"device": "synthetic"}}


def test_writer_requires_the_replay_certificate_filename(tmp_path):
    _, certificate = _packet(tmp_path / "seed")

    with pytest.raises(ValueError, match="certificate.json"):
        write_certificate(
            tmp_path / "packet" / "report.json",
            certificate,
            reproducer_bytes=GENERIC_REPRODUCER,
        )
    assert not (tmp_path / "packet").exists()


def test_writer_replaces_stale_inventory_and_uses_the_current_reproducer(tmp_path):
    certificate_path, certificate = _packet(tmp_path)
    stale = tmp_path / "candidate.capture.json"
    stale.write_bytes(b"stale")
    stale_external = tmp_path / "external" / "source" / "obsolete.py"
    stale_external.write_bytes(b"stale external")
    (tmp_path / "reproducer.py").write_bytes(b"old target reproducer")

    write_certificate(
        certificate_path,
        certificate,
        reproducer_bytes=b"current target reproducer",
    )

    assert stale.read_bytes() == canonical_json_bytes(certificate["candidate"]) + b"\n"
    assert not stale_external.exists()
    assert (tmp_path / "reproducer.py").read_bytes() == b"current target reproducer"
    write_certificate(
        certificate_path,
        certificate,
        reproducer_bytes=b"replacement target reproducer",
    )
    assert (tmp_path / "reproducer.py").read_bytes() == b"replacement target reproducer"


def test_writer_requires_explicit_current_reproducer_bytes(tmp_path):
    certificate_path, certificate = _packet(tmp_path)
    before = certificate_path.read_bytes()
    old_reproducer = (tmp_path / "reproducer.py").read_bytes()

    with pytest.raises(TypeError, match="reproducer_bytes"):
        write_certificate(certificate_path, certificate)

    assert certificate_path.read_bytes() == before
    assert (tmp_path / "reproducer.py").read_bytes() == old_reproducer


@pytest.mark.parametrize(
    "reproducer_bytes", [bytearray(b"not bytes"), b"x" * ((4 << 10) + 1)]
)
def test_writer_rejects_invalid_current_reproducer_bytes_before_promotion(
    tmp_path, reproducer_bytes
):
    certificate_path, certificate = _packet(tmp_path)
    before = certificate_path.read_bytes()

    with pytest.raises(ValueError, match="reproducer"):
        write_certificate(
            certificate_path,
            certificate,
            reproducer_bytes=reproducer_bytes,
        )

    assert certificate_path.read_bytes() == before


@pytest.mark.parametrize(
    "missing_section",
    [
        "candidate",
        "control",
        "analysis",
        "runtime",
        "benchmark",
        "controls",
        "validation",
    ],
)
def test_writer_rejects_missing_certificate_sections_before_staging(
    monkeypatch, tmp_path, missing_section
):
    certificate_path, certificate = _packet(tmp_path)
    before_certificate = certificate_path.read_bytes()
    before_reproducer = (tmp_path / "reproducer.py").read_bytes()
    certificate.pop(missing_section)

    monkeypatch.setattr(
        certificate_module.tempfile,
        "mkdtemp",
        lambda **_kwargs: pytest.fail("malformed certificate allocated staging"),
    )
    with pytest.raises(ValueError, match="certificate sections are incomplete"):
        write_certificate(
            certificate_path,
            certificate,
            reproducer_bytes=b"replacement target reproducer",
        )

    assert certificate_path.read_bytes() == before_certificate
    assert (tmp_path / "reproducer.py").read_bytes() == before_reproducer
    assert list(tmp_path.parent.glob(f".{tmp_path.name}.staging-*")) == []
    assert list(tmp_path.parent.glob(f".{tmp_path.name}.backup-*")) == []


def test_writer_reports_promoted_backup_cleanup_residue(monkeypatch, tmp_path):
    certificate_path, certificate = _packet(tmp_path)
    expected = canonical_json_bytes(certificate) + b"\n"
    real_remove = certificate_module._remove_exact

    def fail_backup_cleanup(path):
        if path.name.startswith(f".{tmp_path.name}.backup-"):
            raise OSError("synthetic backup cleanup failure")
        return real_remove(path)

    monkeypatch.setattr(certificate_module, "_remove_exact", fail_backup_cleanup)
    with pytest.raises(ValueError, match="promotion succeeded; cleanup failed") as first:
        write_certificate(
            certificate_path,
            certificate,
            reproducer_bytes=b"first target reproducer",
        )
    assert certificate_path.read_bytes() == expected
    assert (tmp_path / "reproducer.py").read_bytes() == b"first target reproducer"
    residues = list(tmp_path.parent.glob(f".{tmp_path.name}.backup-*"))
    assert len(residues) == 1
    assert str(residues[0]) in str(first.value)
    assert isinstance(first.value.__cause__, OSError)

    monkeypatch.setattr(
        certificate_module.tempfile,
        "mkdtemp",
        lambda **_kwargs: pytest.fail("retry allocated a new writer staging directory"),
    )
    with pytest.raises(ValueError, match="recoverable residue") as second:
        write_certificate(
            certificate_path,
            certificate,
            reproducer_bytes=b"second target reproducer",
        )
    assert str(residues[0]) in str(second.value)
    assert certificate_path.read_bytes() == expected
    assert (tmp_path / "reproducer.py").read_bytes() == b"first target reproducer"
    assert list(tmp_path.parent.glob(f".{tmp_path.name}.backup-*")) == residues


def test_writer_preserves_pre_promotion_failure_when_staging_cleanup_fails(monkeypatch, tmp_path):
    certificate_path, certificate = _packet(tmp_path)
    before = certificate_path.read_bytes()
    replay_error = ReplayError("synthetic replay rejection")
    real_remove = certificate_module._remove_exact

    def reject(_path):
        raise replay_error

    monkeypatch.setattr(
        certificate_module,
        "replay_certificate",
        reject,
    )

    def fail_staging_cleanup(path):
        if path.name.startswith(f".{tmp_path.name}.staging-"):
            raise OSError("synthetic staging cleanup failure")
        return real_remove(path)

    monkeypatch.setattr(certificate_module, "_remove_exact", fail_staging_cleanup)
    with pytest.raises(ValueError, match="primary failure: synthetic replay rejection") as first:
        write_certificate(
            certificate_path,
            certificate,
            reproducer_bytes=GENERIC_REPRODUCER,
        )
    assert first.value.__cause__ is replay_error
    assert certificate_path.read_bytes() == before
    residues = list(tmp_path.parent.glob(f".{tmp_path.name}.staging-*"))
    assert len(residues) == 1
    assert str(residues[0]) in str(first.value)

    monkeypatch.setattr(
        certificate_module.tempfile,
        "mkdtemp",
        lambda **_kwargs: pytest.fail("retry allocated a new writer staging directory"),
    )
    with pytest.raises(ValueError, match="recoverable residue") as second:
        write_certificate(
            certificate_path,
            certificate,
            reproducer_bytes=GENERIC_REPRODUCER,
        )
    assert str(residues[0]) in str(second.value)
    assert certificate_path.read_bytes() == before
    assert list(tmp_path.parent.glob(f".{tmp_path.name}.staging-*")) == residues


def test_writer_rejects_an_unrelated_packet_root_entry_before_promotion(tmp_path):
    certificate_path, certificate = _packet(tmp_path)
    before = certificate_path.read_bytes()
    sentinel = tmp_path / "unrelated.txt"
    sentinel.write_bytes(b"preserve")

    with pytest.raises(ValueError, match="packet"):
        write_certificate(certificate_path, certificate, reproducer_bytes=GENERIC_REPRODUCER)

    assert certificate_path.read_bytes() == before
    assert sentinel.read_bytes() == b"preserve"


def test_writer_rejects_external_symlink_without_following_it(tmp_path):
    certificate_path, certificate = _packet(tmp_path)
    before = certificate_path.read_bytes()
    source = tmp_path / "external" / "source" / "kernel.py"
    replacement = tmp_path / "replacement.py"
    replacement.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(replacement)

    with pytest.raises(ValueError, match="packet"):
        write_certificate(certificate_path, certificate, reproducer_bytes=GENERIC_REPRODUCER)

    assert certificate_path.read_bytes() == before
    assert source.is_symlink()


def test_writer_rejects_public_replay_before_promotion(monkeypatch, tmp_path):
    certificate_path, certificate = _packet(tmp_path)
    before = certificate_path.read_bytes()
    calls = []

    def reject(path):
        calls.append(path)
        raise ReplayError("synthetic replay rejection")

    monkeypatch.setattr(certificate_module, "replay_certificate", reject)
    with pytest.raises(ValueError, match="packet"):
        write_certificate(certificate_path, certificate, reproducer_bytes=GENERIC_REPRODUCER)

    assert len(calls) == 1
    assert calls[0] != certificate_path
    assert certificate_path.read_bytes() == before


def test_writer_rolls_back_when_packet_promotion_fails(monkeypatch, tmp_path):
    certificate_path, certificate = _packet(tmp_path)
    before = certificate_path.read_bytes()
    packet = certificate_path.parent
    real_replace = Path.replace
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        if Path(destination) == packet and not failed:
            failed = True
            raise OSError("synthetic promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_once)
    with pytest.raises(ValueError, match="packet"):
        write_certificate(certificate_path, certificate, reproducer_bytes=GENERIC_REPRODUCER)

    assert failed
    assert certificate_path.read_bytes() == before


def test_writer_rejects_analysis_digest_drift_before_promotion(tmp_path):
    certificate_path, certificate = _packet(tmp_path)
    before = certificate_path.read_bytes()
    certificate["analysis"]["candidate_digest"] = "0" * 64

    with pytest.raises(ValueError, match="analysis digest"):
        write_certificate(certificate_path, certificate, reproducer_bytes=GENERIC_REPRODUCER)

    assert certificate_path.read_bytes() == before


def test_writer_rejects_semantic_analysis_drift_through_public_replay(tmp_path):
    certificate_path, certificate = _packet(tmp_path)
    before = certificate_path.read_bytes()
    certificate["analysis"]["outcome"] = "VIOLATION"
    certificate["analysis"]["reasons"] = ["BORROWED_COTANGENT_WRITE"]

    with pytest.raises(ValueError, match="packet"):
        write_certificate(certificate_path, certificate, reproducer_bytes=GENERIC_REPRODUCER)

    assert certificate_path.read_bytes() == before


def test_writer_rejects_same_kernel_binding_split_across_source_files(tmp_path):
    _, certificate = _packet(tmp_path / "seed")
    duplicate_file = copy.deepcopy(certificate["target"]["files"][0])
    duplicate_file["packet_path"] = "external/source/other.py"
    duplicate_file["target_path"] = "src/other.py"
    certificate["target"]["files"].append(duplicate_file)
    duplicate_kernel = copy.deepcopy(certificate["target"]["kernels"][0])
    duplicate_kernel["file"] = duplicate_file["packet_path"]
    certificate["target"]["kernels"].append(duplicate_kernel)

    with pytest.raises(ValueError, match="kernel binding"):
        write_certificate(
            tmp_path / "duplicate" / "certificate.json",
            certificate,
            reproducer_bytes=GENERIC_REPRODUCER,
        )
    assert not (tmp_path / "duplicate").exists()


def test_writer_rejects_empty_kernel_manifest_before_staging(tmp_path):
    _, certificate = _packet(tmp_path / "seed")
    certificate["target"]["kernels"] = []
    destination = tmp_path / "empty" / "certificate.json"

    with pytest.raises(ValueError, match="kernel manifest"):
        write_certificate(
            destination,
            certificate,
            reproducer_bytes=GENERIC_REPRODUCER,
        )

    assert not destination.parent.exists()


def test_writer_rejects_unknown_mapping_schema_before_mapping_methods(tmp_path):
    class HostileDict(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("caller get must not run")

        def items(self):
            raise AssertionError("caller items must not run")

    path = tmp_path / "not-created" / "certificate.json"
    with pytest.raises(ValueError, match="unsupported certificate schema"):
        write_certificate(
            path,
            HostileDict({"schema": "unknown"}),
            reproducer_bytes=GENERIC_REPRODUCER,
        )
    assert not path.parent.exists()


def test_v2_writer_snapshots_a_recognized_mapping_before_target_validation(tmp_path):
    class HostileDict(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("caller get must not run")

    path = tmp_path / "not-created" / "certificate.json"
    with pytest.raises(ValueError, match="certificate target"):
        write_certificate(
            path,
            HostileDict({"schema": CERTIFICATE_SCHEMA}),
            reproducer_bytes=GENERIC_REPRODUCER,
        )
    assert not path.parent.exists()
