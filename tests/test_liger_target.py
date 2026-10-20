from __future__ import annotations

import builtins
import copy
import hashlib
import importlib
import io
import json
import os
import py_compile
import runpy
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from daggertrace import liger_packet
from daggertrace import source_files
from daggertrace import liger_target as targets
from daggertrace import target_support
from daggertrace.canonical import sha256_json, write_canonical_json
from daggertrace.cli import replay_certificate
from daggertrace.errors import TargetError
from daggertrace.model import Capture
from daggertrace.ownership import analyze
from daggertrace.replay import ReplayError
from daggertrace.replay_packet import _runtime_evidence as replay_runtime_evidence
from daggertrace.source_files import (
    MAX_PINNED_FILE_BYTES,
    _read_regular_file,
    _validated_tree_file_path,
)
from daggertrace.liger_target import audit_liger
from daggertrace.targets import run_liger_reproducer


_CHECKED_LIGER_PACKET = (
    Path(__file__).parents[1] / "evidence" / "results" / "bundle" / "liger" / "liger-rmsnorm"
)


def _checked_liger_certificate():
    return json.loads((_CHECKED_LIGER_PACKET / "certificate.json").read_text())


def _set_liger_partial_rows(capture, allocation_index, result_index, hop_index, partial_rows):
    nodes = capture["backward"]["nodes"]
    nodes[allocation_index]["args"]["tuple"][0][0] = partial_rows
    nodes[allocation_index]["tensor"]["shape"][0] = partial_rows
    nodes[result_index]["tensor"]["shape"][0] = partial_rows
    hop = nodes[hop_index]["kwargs"]
    hop["constants"]["rows_per_program"] = (targets.SHAPE[0] + partial_rows - 1) // partial_rows
    hop["grid"] = [{"tuple": [partial_rows, 1, 1]}]


def _append_unreachable_liger_add(capture, graph):
    nodes = capture[graph]["nodes"]
    output = nodes.pop()
    extra = copy.deepcopy(nodes[-1])
    extra["id"] = output["id"]
    extra["gradient_accumulation"] = False
    extra["kwargs"] = {}
    extra["tensor"]["alias_group"] = f"extra-{graph}"
    output["id"] = f"n{len(nodes) + 1}"
    nodes.extend((extra, output))


def _write_resealed_liger_packet(packet, certificate):
    certificate["analysis"] = analyze(
        Capture.from_dict(certificate["candidate"]),
        Capture.from_dict(certificate["control"]),
    ).to_dict()
    for filename, key in (
        ("candidate.capture.json", "candidate"),
        ("control.capture.json", "control"),
    ):
        write_canonical_json(packet / filename, certificate[key])
    certificate["digest"] = sha256_json(
        {key: value for key, value in certificate.items() if key != "digest"}
    )
    write_canonical_json(packet / "certificate.json", certificate)


def _delta(*, output=0.0, gx=0.0, gw=0.0):
    return {"output_max_abs": output, "gx_max_abs": gx, "gw_max_abs": gw}


def _generic_liger_result():
    return {
        "valid": True,
        "outcome": "VIOLATION",
        "reasons": ["BORROWED_COTANGENT_WRITE", "CONTRIBUTION_COLLAPSE"],
        "borrowed_writes": 1,
        "contextual_collapse": True,
    }


def test_pinned_source_read_rejects_oversized_regular_file_before_loading(tmp_path):
    path = tmp_path / "oversized.py"
    with path.open("wb") as handle:
        handle.truncate(MAX_PINNED_FILE_BYTES + 1)

    with pytest.raises(TargetError, match="exceeds .* bytes"):
        _read_regular_file(path, "oversized.py", "fixture")


@pytest.mark.parametrize(
    "relative",
    (
        "/absolute",
        "//server/share",
        r"\\server\share",
        "foo//bar",
        "foo/./bar",
        "foo/../bar",
        "C:foo",
        "C:/foo",
        "foo/C:bar",
        "foo\x00bar",
        "\udcff",
    ),
)
def test_pinned_tree_path_rejects_noncanonical_and_windows_forms(tmp_path, relative):
    root = tmp_path / "tree"
    root.mkdir()

    with pytest.raises(TargetError, match="invalid pinned fixture tree path"):
        _validated_tree_file_path(root, relative, "fixture")


def test_pinned_tree_path_accepts_canonical_unicode_path(tmp_path):
    root = tmp_path / "tree"
    path = root / "src" / "café.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"source\n")

    assert _validated_tree_file_path(root, "src/café.py", "fixture") == path


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX byte filesystem paths")
def test_pinned_tree_reads_unicode_filename_under_ascii_locale(tmp_path):
    root = tmp_path / "tree"
    source = root / "café.py"
    root.mkdir()
    source.write_bytes(b"source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    script = "\n".join(
        (
            "from pathlib import Path",
            "from daggertrace.source_files import _read_verified_tree_file",
            f"value = _read_verified_tree_file(Path({json.dumps(str(root), ensure_ascii=True)}), 'caf\\u00e9.py', '{digest}', 'fixture')",
            "print(value.decode())",
        )
    )
    environment = {
        **os.environ,
        "LC_ALL": "C",
        "PYTHONCOERCECLOCALE": "0",
        "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        "PYTHONUTF8": "0",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        env=environment,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "source\n"


def _valid_runtime_evidence():
    return {
        "candidate_vs_reference": _delta(gx=0.5),
        "control_vs_reference": _delta(gx=1e-6),
        "candidate_vs_control": _delta(gx=0.5),
        "cotangent": {
            "candidate_delta": 0.5,
            "control_delta": 0.0,
            "candidate_version": [0, 0],
            "control_version": [0, 0],
        },
        "compiled": {
            "candidate_vs_reference": _delta(gx=0.5),
            "control_vs_reference": _delta(gx=1e-6),
            "candidate_eager_vs_compiled": _delta(),
            "control_eager_vs_compiled": _delta(),
            "candidate_observation": {
                "cotangent_delta": 0.0,
                "cotangent_version": [0, 0],
            },
            "control_observation": {
                "cotangent_delta": 0.0,
                "cotangent_version": [0, 0],
            },
        },
    }


def _valid_compile_evidence():
    return {
        "cold_ms": 1.0,
        "generated_kernel_count": 4,
        "ir_nodes_pre_fusion": 9,
        "counter_delta": {
            "aot_autograd": {"total": 1, "ok": 1, "autograd_cache_bypass": 1},
            "inductor": {"extern_calls": 2},
        },
    }


def _runtime_environment():
    return {
        "python": "3.14.6",
        "platform": "test",
        "torch": "2.12.1+cu130",
        "torch_git": "7269437d655783a26cba32aa88195b741ff496aa",
        "triton": "3.7.1",
        "cuda_runtime": "13.0",
        "device": "test-device",
        "capability": [1, 0],
        "total_memory_bytes": 0,
        "multiprocessors": 0,
        "driver": None,
    }


def _valid_liger_runtime_witness():
    witness = _valid_runtime_evidence()
    witness["peak_allocated_bytes"] = {"reference": 0, "candidate": 1, "control": 0}
    witness["hashes"] = {
        "reference_gx": "a" * 64,
        "candidate_gx": "b" * 64,
        "control_gx": "c" * 64,
    }
    witness["compiled"]["compile"] = {
        "candidate": _valid_compile_evidence(),
        "control": _valid_compile_evidence(),
    }
    witness["control_witness"] = {
        "isolated": {"candidate_vs_control": _delta()},
        "scaled": {"candidate_vs_control": _delta()},
    }
    witness["environment"] = _runtime_environment()
    witness["shape"] = [2, 128]
    witness["seed"] = 7
    return witness


def test_liger_runtime_evidence_emits_generic_paired_occurrences(monkeypatch):
    occurrence = _valid_compile_evidence()
    comparison = _valid_liger_runtime_witness()
    compile_results = iter(
        [
            ("candidate", object(), occurrence),
            ("control", object(), occurrence),
        ]
    )
    monkeypatch.setattr(targets, "_run_lane", lambda *_args: object())
    monkeypatch.setattr(targets, "_comparison", lambda *_args: copy.deepcopy(comparison))
    monkeypatch.setattr(targets, "_compile_lane", lambda *_args: next(compile_results))
    monkeypatch.setattr(targets, "_lane_delta", lambda *_args: _delta())
    monkeypatch.setattr(
        targets,
        "_lane_observation",
        lambda *_args: {"cotangent_delta": 0.0, "cotangent_version": [0, 0]},
    )
    monkeypatch.setattr(targets, "_validate_runtime_evidence", lambda _comparison: None)
    monkeypatch.setattr(targets, "_gpu_environment", lambda _target_name: _runtime_environment())

    runtime, target_runtime, _compiled = targets._runtime_evidence_pair(
        object(), object(), object(), object(), object(), object()
    )

    replay_runtime_evidence(runtime)
    assert runtime == {
        "environment": _runtime_environment(),
        "candidate": {"compiled": {"occurrence": occurrence}},
        "control": {"compiled": {"occurrence": occurrence}},
    }
    assert target_runtime["compiled"]["compile"] == {
        "candidate": occurrence,
        "control": occurrence,
    }


def _replace_path(value, path, replacement):
    owner = value
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = replacement


def test_runtime_gate_accepts_complete_witness():
    targets._validate_runtime_evidence(_valid_runtime_evidence())


@pytest.mark.parametrize(
    "path",
    (
        ("candidate_vs_reference", "output_max_abs"),
        ("candidate_vs_reference", "gw_max_abs"),
        ("control_vs_reference", "output_max_abs"),
        ("control_vs_reference", "gx_max_abs"),
        ("control_vs_reference", "gw_max_abs"),
        ("compiled", "candidate_vs_reference", "output_max_abs"),
        ("compiled", "candidate_vs_reference", "gw_max_abs"),
        ("compiled", "control_vs_reference", "output_max_abs"),
        ("compiled", "control_vs_reference", "gx_max_abs"),
        ("compiled", "control_vs_reference", "gw_max_abs"),
    ),
)
@pytest.mark.parametrize("replacement", (float("nan"), float("inf"), -1.0))
def test_runtime_gate_rejects_nonfinite_or_negative_upper_deltas(path, replacement):
    evidence = _valid_runtime_evidence()
    _replace_path(evidence, path, replacement)

    with pytest.raises(TargetError):
        targets._validate_runtime_evidence(evidence)


@pytest.mark.parametrize("value", (0.0, 2e-5))
def test_runtime_gate_accepts_zero_and_tolerance_upper_deltas(value):
    evidence = _valid_runtime_evidence()
    for path in (
        ("candidate_vs_reference", "output_max_abs"),
        ("candidate_vs_reference", "gw_max_abs"),
        ("control_vs_reference", "output_max_abs"),
        ("control_vs_reference", "gx_max_abs"),
        ("control_vs_reference", "gw_max_abs"),
        ("compiled", "candidate_vs_reference", "output_max_abs"),
        ("compiled", "candidate_vs_reference", "gw_max_abs"),
        ("compiled", "control_vs_reference", "output_max_abs"),
        ("compiled", "control_vs_reference", "gx_max_abs"),
        ("compiled", "control_vs_reference", "gw_max_abs"),
    ):
        _replace_path(evidence, path, value)

    targets._validate_runtime_evidence(evidence)


@pytest.mark.parametrize(
    "path",
    (
        ("candidate_vs_reference", "gx_max_abs"),
        ("compiled", "candidate_vs_reference", "gx_max_abs"),
        ("cotangent", "candidate_delta"),
    ),
)
def test_runtime_gate_rejects_infinite_lower_bound_witnesses(path):
    evidence = _valid_runtime_evidence()
    _replace_path(evidence, path, float("inf"))

    with pytest.raises(TargetError):
        targets._validate_runtime_evidence(evidence)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("cotangent", "candidate_delta"), 1e-4),
        (("cotangent", "control_delta"), 1e-8),
        (("cotangent", "candidate_version"), [0, 1]),
        (("cotangent", "control_version"), [0, 1]),
        (("candidate_vs_control", "output_max_abs"), 1e-8),
        (("candidate_vs_control", "gw_max_abs"), 1e-8),
        (("compiled", "candidate_eager_vs_compiled", "output_max_abs"), 1e-8),
        (("compiled", "candidate_eager_vs_compiled", "gx_max_abs"), 1e-8),
        (("compiled", "candidate_eager_vs_compiled", "gw_max_abs"), 1e-8),
        (("compiled", "control_eager_vs_compiled", "output_max_abs"), 1e-8),
        (("compiled", "control_eager_vs_compiled", "gx_max_abs"), 1e-8),
        (("compiled", "control_eager_vs_compiled", "gw_max_abs"), 1e-8),
        (("compiled", "candidate_observation", "cotangent_delta"), 1e-8),
        (("compiled", "control_observation", "cotangent_delta"), 1e-8),
        (("compiled", "candidate_observation", "cotangent_version"), [0, 1]),
        (("compiled", "control_observation", "cotangent_version"), [0, 1]),
    ],
)
def test_runtime_gate_rejects_incomplete_evidence(path, replacement):
    evidence = copy.deepcopy(_valid_runtime_evidence())
    _replace_path(evidence, path, replacement)
    with pytest.raises(TargetError):
        targets._validate_runtime_evidence(evidence)


def test_compile_gate_accepts_exact_lane_metrics():
    targets._validate_compile_evidence(_valid_compile_evidence())


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("counter_delta", "aot_autograd", "total"), 2),
        (("counter_delta", "aot_autograd", "ok"), 0),
        (("counter_delta", "aot_autograd", "not_ok"), 1),
        (("counter_delta", "inductor", "extern_calls"), 1),
        (("generated_kernel_count",), 3),
        (("ir_nodes_pre_fusion",), 8),
    ],
)
def test_compile_gate_rejects_noncanonical_lane_metrics(path, replacement):
    evidence = copy.deepcopy(_valid_compile_evidence())
    _replace_path(evidence, path, replacement)
    with pytest.raises(TargetError):
        targets._validate_compile_evidence(evidence)


class _Response:
    def __init__(self, payload, headers=None):
        self.payload = payload
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def read(self, _limit):
        return self.payload


def _archive_payload():
    payload = io.BytesIO()
    prefix = f"Liger-Kernel-{targets.LIGER_REVISION}/"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(f"{prefix}LICENSE", "license")
    return payload.getvalue()


def _exact_tree_fixture(tmp_path):
    root = tmp_path / "Liger-Kernel"
    checked_tree = _CHECKED_LIGER_PACKET / "external" / "liger" / "tree"
    for relative, _expected, _role in liger_packet._selected_liger_files():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((checked_tree / relative).read_bytes())
    return root


@pytest.mark.parametrize(
    "relative",
    [
        liger_packet.LIGER_SOURCE_PATH,
        liger_packet.LIGER_UTILS_PATH,
        liger_packet.LIGER_ROOT_UTILS_PATH,
        liger_packet.LIGER_NOTICE_PATH,
        "LICENSE",
        liger_packet.LIGER_TRITON_LICENSE_PATH,
    ],
)
def test_liger_tree_requires_selected_files_to_be_regular(tmp_path, relative):
    root = _exact_tree_fixture(tmp_path)
    provenance = root / relative
    provenance.unlink()
    provenance.symlink_to(root / "LICENSE")

    with pytest.raises(TargetError, match="not a regular file"):
        liger_packet._validate_liger_tree(root)


@pytest.mark.parametrize(
    "relative",
    [
        liger_packet.LIGER_SOURCE_PATH,
        liger_packet.LIGER_UTILS_PATH,
        liger_packet.LIGER_ROOT_UTILS_PATH,
        liger_packet.LIGER_NOTICE_PATH,
        "LICENSE",
        liger_packet.LIGER_TRITON_LICENSE_PATH,
    ],
)
def test_liger_tree_requires_exact_selected_file_hash(tmp_path, relative):
    root = _exact_tree_fixture(tmp_path)
    path = root / relative
    path.write_bytes(path.read_bytes() + b"\x00")

    with pytest.raises(TargetError, match=f"hash mismatch for {relative}"):
        liger_packet._validate_liger_tree(root)


def test_liger_tree_rejects_a_symlinked_source_directory(tmp_path):
    root = _exact_tree_fixture(tmp_path)
    source_directory = root / "src" / "liger_kernel" / "ops"
    shutil.rmtree(source_directory)
    outside = tmp_path / "outside"
    outside.mkdir()
    source_directory.symlink_to(outside, target_is_directory=True)

    with pytest.raises(TargetError, match="path component is not a directory"):
        liger_packet._validate_liger_tree(root)


def test_liger_tree_rejects_a_symlinked_root(tmp_path):
    root = _exact_tree_fixture(tmp_path)
    alias = tmp_path / "target-alias"
    alias.symlink_to(root, target_is_directory=True)

    with pytest.raises(TargetError, match="tree is not a directory"):
        liger_packet._validate_liger_tree(alias)


def _liger_kernel_capture():
    return SimpleNamespace(
        kernels=(
            SimpleNamespace(
                module="liger_kernel.ops.rms_norm",
                qualname="_rms_norm_backward_kernel",
                source_sha256="a" * 64,
            ),
            SimpleNamespace(
                module="liger_kernel.ops.rms_norm",
                qualname="_rms_norm_forward_kernel",
                source_sha256="b" * 64,
            ),
        )
    )


def test_liger_target_manifest_binds_the_full_packet_closure(tmp_path):
    root = _exact_tree_fixture(tmp_path)
    liger_packet._validate_liger_tree(root)

    candidate = _liger_kernel_capture()
    control = _liger_kernel_capture()
    output = tmp_path / "packet"
    liger_packet._stage_external_sources(root, output)
    identity = targets._target_identity(root, candidate, output, control=control)

    assert set(identity) == {"repository", "revision", "files", "kernels", "specialization"}
    expected_files = [
        {
            "packet_path": f"external/liger/tree/{relative}",
            "target_path": relative,
            "size": len((output / "external" / "liger" / "tree" / relative).read_bytes()),
            "sha256": expected,
            "role": role,
        }
        for relative, expected, role in liger_packet._selected_liger_files()
    ]
    notice_path = output / "external" / "liger" / "NOTICE.json"
    unsloth_path = output / "external" / "liger" / "UNSLOTH-APACHE-2.0.txt"
    expected_files.extend(
        [
            {
                "packet_path": "external/liger/NOTICE.json",
                "target_path": "NOTICE.json",
                "size": notice_path.stat().st_size,
                "sha256": hashlib.sha256(notice_path.read_bytes()).hexdigest(),
                "role": "notice",
            },
            {
                "packet_path": "external/liger/UNSLOTH-APACHE-2.0.txt",
                "target_path": "licenses/UNSLOTH-APACHE-2.0.txt",
                "size": unsloth_path.stat().st_size,
                "sha256": hashlib.sha256(unsloth_path.read_bytes()).hexdigest(),
                "role": "license",
            },
        ]
    )
    assert identity["files"] == expected_files
    assert identity["kernels"] == [
        {
            "module": "liger_kernel.ops.rms_norm",
            "qualname": "_rms_norm_backward_kernel",
            "file": "external/liger/tree/src/liger_kernel/ops/rms_norm.py",
            "source_sha256": "a" * 64,
        },
        {
            "module": "liger_kernel.ops.rms_norm",
            "qualname": "_rms_norm_forward_kernel",
            "file": "external/liger/tree/src/liger_kernel/ops/rms_norm.py",
            "source_sha256": "b" * 64,
        },
    ]
    assert identity["specialization"] == {
        "shape": [2, 128],
        "dtype": "float32",
        "device": "cuda",
        "casting_mode": "gemma",
        "offset": 1.0,
        "epsilon": 1e-6,
        "elementwise_affine": True,
        "row_mode": True,
        "candidate_in_place": True,
        "control_in_place": False,
        "expression": "x + rms_norm(x)",
    }

    staged = output / "external" / "liger" / "tree"
    for relative, _expected, _role in liger_packet._selected_liger_files():
        assert (staged / relative).read_bytes() == (root / relative).read_bytes()
    unsloth = output / "external" / "liger" / "UNSLOTH-APACHE-2.0.txt"
    assert (
        targets.sha256_bytes(unsloth.read_bytes())
        == liger_packet.UNSLOTH_LICENSE_SHA256
    )
    assert json.loads((output / "external" / "liger" / "NOTICE.json").read_text()) == {
        "repository": targets.LIGER_REPOSITORY,
        "revision": targets.LIGER_REVISION,
        "source_header_contains_upstream_attribution": True,
        "upstream": (
            "https://github.com/unslothai/unsloth/tree/"
            f"{liger_packet.UNSLOTH_REVISION}"
        ),
        "license_sha256": liger_packet.UNSLOTH_LICENSE_SHA256,
    }
    assert {
        path.relative_to(output)
        for path in output.rglob("*")
        if path.is_file()
    } == {
        Path("external/liger/NOTICE.json"),
        Path("external/liger/UNSLOTH-APACHE-2.0.txt"),
        Path("external/liger/tree/LICENSE"),
        Path("external/liger/tree/NOTICE"),
        Path("external/liger/tree/licenses/LICENSE-MIT-triton"),
        Path("external/liger/tree/src/liger_kernel/ops/rms_norm.py"),
        Path("external/liger/tree/src/liger_kernel/ops/utils.py"),
        Path("external/liger/tree/src/liger_kernel/utils.py"),
    }


def test_liger_target_identity_rejects_a_kernel_set_outside_fixed_row_mode(tmp_path):
    root = _exact_tree_fixture(tmp_path)
    output = tmp_path / "packet"
    liger_packet._stage_external_sources(root, output)
    candidate = _liger_kernel_capture()
    candidate.kernels = (
        *candidate.kernels,
        SimpleNamespace(
            module="liger_kernel.ops.rms_norm",
            qualname="_rms_norm_block_forward_kernel",
            source_sha256="c" * 64,
        ),
    )

    with pytest.raises(TargetError, match="fixed row-mode specialization"):
        targets._target_identity(
            root,
            candidate,
            output,
            control=_liger_kernel_capture(),
        )


def test_liger_replay_rejects_resealed_target_specialization(monkeypatch, tmp_path):
    root = _exact_tree_fixture(tmp_path)
    candidate = _liger_kernel_capture()
    packet = tmp_path / "packet"
    liger_packet._stage_external_sources(root, packet)
    runtime_witness = _valid_liger_runtime_witness()
    target = targets._target_identity(
        root,
        candidate,
        packet,
        control=candidate,
        runtime_witness=runtime_witness,
    )
    certificate = {
        "target": copy.deepcopy(target),
        "candidate": {},
        "control": {},
        "runtime": targets._generic_liger_runtime(runtime_witness),
    }
    monkeypatch.setattr(
        targets,
        "Capture",
        SimpleNamespace(from_dict=lambda _raw: candidate),
    )
    monkeypatch.setattr(
        targets, "_validate_liger_capture_specialization", lambda *_args, **_kwargs: None
    )
    certificate_path = packet / "certificate.json"

    result = targets._replay_liger_loaded(certificate, certificate_path, _generic_liger_result())

    assert result["adapter"] == "liger-rmsnorm"
    certificate["target"]["specialization"]["expression"] = "resealed"
    with pytest.raises(TargetError, match="target identity"):
        targets._replay_liger_loaded(certificate, certificate_path, _generic_liger_result())
    certificate["target"]["specialization"]["expression"] = "x + rms_norm(x)"
    certificate["target"]["specialization"]["runtime_witness"]["shape"][0] = 3
    with pytest.raises(TargetError, match="runtime witness shape changed"):
        targets._replay_liger_loaded(certificate, certificate_path, _generic_liger_result())
    certificate["target"]["specialization"]["runtime_witness"]["shape"][0] = 2
    certificate["target"]["specialization"]["runtime_witness"]["control_witness"]["isolated"][
        "candidate_vs_control"
    ]["gx_max_abs"] = 1e-6
    with pytest.raises(TargetError, match="isolated control witness changed"):
        targets._replay_liger_loaded(certificate, certificate_path, _generic_liger_result())


def test_liger_capture_specialization_binds_checked_packet_and_device_partial_rows():
    certificate = _checked_liger_certificate()

    targets._validate_liger_capture_specialization(
        Capture.from_dict(certificate["candidate"]), Capture.from_dict(certificate["control"])
    )

    candidate = copy.deepcopy(certificate["candidate"])
    control = copy.deepcopy(certificate["control"])

    _set_liger_partial_rows(candidate, 5, 8, 6, 80)
    _set_liger_partial_rows(control, 5, 9, 7, 80)
    targets._validate_liger_capture_specialization(
        Capture.from_dict(candidate), Capture.from_dict(control), expected_partial_rows=80
    )

    with pytest.raises(TargetError, match="partial rows differ from its runtime witness"):
        targets._validate_liger_capture_specialization(
            Capture.from_dict(candidate), Capture.from_dict(control), expected_partial_rows=36
        )

    _set_liger_partial_rows(candidate, 5, 8, 6, 1)
    _set_liger_partial_rows(control, 5, 9, 7, 1)
    targets._validate_liger_capture_specialization(
        Capture.from_dict(candidate), Capture.from_dict(control), expected_partial_rows=1
    )

    _set_liger_partial_rows(control, 5, 9, 7, 2)
    with pytest.raises(TargetError, match="candidate/control partial row counts differ"):
        targets._validate_liger_capture_specialization(
            Capture.from_dict(candidate), Capture.from_dict(control)
        )

    _set_liger_partial_rows(candidate, 5, 8, 6, 80)
    _set_liger_partial_rows(control, 5, 9, 7, 80)
    for capture, hop_index in ((candidate, 6), (control, 7)):
        capture["backward"]["nodes"][hop_index]["kwargs"]["constants"]["rows_per_program"] = 0
    with pytest.raises(TargetError, match="backward specialization changed"):
        targets._validate_liger_capture_specialization(
            Capture.from_dict(candidate), Capture.from_dict(control)
        )


def test_liger_audit_binds_capture_partial_rows_to_runtime_witness(monkeypatch, tmp_path):
    certificate = _checked_liger_certificate()
    candidate = copy.deepcopy(certificate["candidate"])
    control = copy.deepcopy(certificate["control"])
    _set_liger_partial_rows(candidate, 5, 8, 6, 80)
    _set_liger_partial_rows(control, 5, 9, 7, 80)
    captures = iter((Capture.from_dict(candidate), Capture.from_dict(control)))
    runtime_witness = _valid_liger_runtime_witness()
    runtime_witness["environment"]["multiprocessors"] = 36
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        float32=object(),
        manual_seed=lambda _seed: None,
        randn=lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        targets,
        "_require_module",
        lambda name, _display_name: fake_torch if name == "torch" else object(),
    )
    monkeypatch.setattr(targets, "_require_exact_gpu_framework", lambda *_args: None)
    monkeypatch.setattr(targets, "_load_liger_function", lambda _root: object())
    monkeypatch.setattr(targets, "_liger_function", lambda *_args: object())
    monkeypatch.setattr(targets, "_capture", lambda *_args, **_kwargs: next(captures))
    monkeypatch.setattr(
        targets._ownership,
        "analyze",
        lambda *_args: SimpleNamespace(
            outcome=SimpleNamespace(value="VIOLATION"), collapse=object(), to_dict=lambda: {}
        ),
    )
    monkeypatch.setattr(targets, "_reference_function", lambda: object())
    monkeypatch.setattr(
        targets,
        "_runtime_evidence_pair",
        lambda *_args: ({}, copy.deepcopy(runtime_witness), {}),
    )
    monkeypatch.setattr(
        targets,
        "_control_evidence",
        lambda *_args: ({}, copy.deepcopy(runtime_witness["control_witness"])),
    )
    monkeypatch.setattr(
        targets,
        "_benchmark",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("benchmark reached")),
    )

    with pytest.raises(TargetError, match="capture partial rows differ from its runtime witness"):
        audit_liger(tmp_path / "audit", target_root=tmp_path)


@pytest.mark.parametrize(
    ("lane", "path", "replacement", "message"),
    [
        (
            "candidate",
            ("diagnostics", "inputs", 0, "dtype"),
            "float16",
            "candidate input metadata specialization changed",
        ),
        (
            "candidate",
            ("forward", "nodes", 3, "tensor", "dtype"),
            "float16",
            "candidate forward specialization changed",
        ),
        (
            "candidate",
            ("forward", "nodes", 5, "kwargs", "constants", "offset"),
            0.0,
            "candidate forward specialization changed",
        ),
        (
            "candidate",
            ("backward", "nodes", 6, "kwargs", "grid"),
            [{"tuple": [2, 1, 1]}],
            "candidate backward specialization changed",
        ),
        (
            "candidate",
            ("backward", "nodes", 6, "kwargs", "bindings", "dX_ptr"),
            {"ref": "n5"},
            "candidate gradient semantics changed",
        ),
        (
            "candidate",
            ("forward", "nodes", 9, "args", "tuple"),
            [{"ref": "n8"}, {"ref": "n0"}],
            "candidate forward output semantics changed",
        ),
        (
            "candidate",
            ("forward", "nodes", 9, "kwargs"),
            {"alpha": 2},
            "candidate forward output semantics changed",
        ),
        (
            "candidate",
            ("backward", "nodes", 13, "kwargs"),
            {"alpha": 2},
            "candidate gradient semantics changed",
        ),
        (
            "control",
            ("backward", "nodes", 7, "kwargs", "bindings", "dX_ptr"),
            {"ref": "n4"},
            "control gradient destination changed",
        ),
        (
            "control",
            ("backward", "nodes", 12, "kwargs"),
            {"alpha": 2},
            "control gradient semantics changed",
        ),
    ],
)
def test_liger_capture_specialization_rejects_resealed_semantic_changes(
    lane, path, replacement, message
):
    certificate = _checked_liger_certificate()
    candidate = copy.deepcopy(certificate["candidate"])
    control = copy.deepcopy(certificate["control"])
    changed = candidate if lane == "candidate" else control
    parent = changed
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = replacement

    with pytest.raises(TargetError, match=message):
        targets._validate_liger_capture_specialization(
            Capture.from_dict(candidate), Capture.from_dict(control)
        )


@pytest.mark.parametrize(
    ("lane", "graph"),
    [
        ("candidate", "forward"),
        ("candidate", "backward"),
        ("control", "forward"),
        ("control", "backward"),
    ],
)
def test_liger_capture_specialization_rejects_extra_graph_nodes(lane, graph):
    certificate = _checked_liger_certificate()
    candidate = copy.deepcopy(certificate["candidate"])
    control = copy.deepcopy(certificate["control"])
    _append_unreachable_liger_add(candidate if lane == "candidate" else control, graph)

    with pytest.raises(TargetError, match=f"Liger {lane} graph topology changed"):
        targets._validate_liger_capture_specialization(
            Capture.from_dict(candidate), Capture.from_dict(control)
        )


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (
            ("diagnostics", "inputs", 0, "dtype"),
            "float16",
            "Liger candidate input metadata specialization changed",
        ),
        (
            ("forward", "nodes", 9, "gradient_accumulation"),
            True,
            "Liger candidate forward output semantics changed",
        ),
    ],
)
def test_liger_public_replay_rejects_resealed_specialization(
    tmp_path, path, replacement, message
):
    packet = tmp_path / "packet"
    shutil.copytree(_CHECKED_LIGER_PACKET, packet)
    certificate_path = packet / "certificate.json"
    certificate = json.loads(certificate_path.read_text())

    for lane in ("candidate", "control"):
        changed = certificate[lane]
        for key in path[:-1]:
            changed = changed[key]
        changed[path[-1]] = copy.deepcopy(replacement)
    _write_resealed_liger_packet(packet, certificate)

    with pytest.raises(ReplayError, match=message):
        replay_certificate(certificate_path)


@pytest.mark.parametrize(
    ("lanes", "graph", "expected_lane"),
    [
        (("candidate", "control"), "forward", "candidate"),
        (("candidate",), "backward", "candidate"),
        (("control",), "backward", "control"),
    ],
)
def test_liger_public_replay_rejects_resealed_extra_graph_node(
    tmp_path, lanes, graph, expected_lane
):
    packet = tmp_path / "packet"
    shutil.copytree(_CHECKED_LIGER_PACKET, packet)
    certificate = json.loads((packet / "certificate.json").read_text())
    for lane in lanes:
        _append_unreachable_liger_add(certificate[lane], graph)
    _write_resealed_liger_packet(packet, certificate)

    with pytest.raises(ReplayError, match=f"Liger {expected_lane} graph topology changed"):
        replay_certificate(packet / "certificate.json")


def test_staging_revalidates_the_transitive_source_closure(tmp_path):
    root = _exact_tree_fixture(tmp_path)
    (root / liger_packet.LIGER_UTILS_PATH).write_text("changed", encoding="utf-8")

    with pytest.raises(TargetError, match="hash mismatch for src/liger_kernel/ops/utils.py"):
        liger_packet._stage_external_sources(root, tmp_path / "packet")

    assert not (tmp_path / "packet").exists()


def test_fetch_wraps_invalid_content_length(monkeypatch, tmp_path):
    monkeypatch.setattr(
        target_support.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(b"", {"Content-Length": "invalid"}),
    )
    with pytest.raises(TargetError) as raised:
        targets.fetch_liger(tmp_path / "cache")
    assert isinstance(raised.value.__cause__, ValueError)


def test_fetch_wraps_bad_zip(monkeypatch, tmp_path):
    monkeypatch.setattr(
        target_support.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(b"not a zip"),
    )
    with pytest.raises(TargetError) as raised:
        targets.fetch_liger(tmp_path / "cache")
    assert isinstance(raised.value.__cause__, target_support.zipfile.BadZipFile)


def test_fetch_wraps_os_error(monkeypatch, tmp_path):
    failure = OSError("offline")

    def unavailable(_request, timeout):
        raise failure

    monkeypatch.setattr(target_support.urllib.request, "urlopen", unavailable)
    with pytest.raises(TargetError) as raised:
        targets.fetch_liger(tmp_path / "cache")
    assert raised.value.__cause__ is failure


def test_fetch_wraps_extraction_os_error(monkeypatch, tmp_path):
    failure = OSError("read-only cache")

    def cannot_extract(_archive, _destination):
        raise failure

    monkeypatch.setattr(
        target_support.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(_archive_payload()),
    )
    monkeypatch.setattr(target_support.zipfile.ZipFile, "extractall", cannot_extract)
    with pytest.raises(TargetError) as raised:
        targets.fetch_liger(tmp_path / "cache")
    assert raised.value.__cause__ is failure


def test_fetch_validation_failure_cleans_the_exact_staging_tree(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setattr(
        target_support.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(_archive_payload()),
    )

    def reject_staged_tree(root):
        assert root.parent.parent == cache
        assert root.parent.name.startswith(f".{root.name}.staging-")
        raise TargetError("invalid staged target")

    monkeypatch.setattr(liger_packet, "_validate_liger_tree", reject_staged_tree)
    with pytest.raises(TargetError, match="invalid staged target"):
        targets.fetch_liger(cache)

    assert cache.is_dir()
    assert list(cache.iterdir()) == []


@pytest.mark.parametrize(
    "primary_type",
    (TargetError, AssertionError),
    ids=("target-error", "programmer-error"),
)
def test_fetch_preserves_primary_failure_when_staging_cleanup_fails(
    monkeypatch, tmp_path, primary_type
):
    cache = tmp_path / "cache"
    primary_error = primary_type("invalid staged target")
    primary_cause = OSError("validator cause")
    cleanup_error = OSError("cleanup failed")
    staging_paths = []
    cleanup_paths = []
    monkeypatch.setattr(
        target_support.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(_archive_payload()),
    )

    def reject_staged_tree(root):
        staging_paths.append(root.parent)
        raise primary_error from primary_cause

    def fail_cleanup(path):
        cleanup_paths.append(path)
        raise cleanup_error

    monkeypatch.setattr(liger_packet, "_validate_liger_tree", reject_staged_tree)
    monkeypatch.setattr(target_support.shutil, "rmtree", fail_cleanup)
    with pytest.raises(primary_type) as raised:
        targets.fetch_liger(cache)

    assert raised.value is primary_error
    assert raised.value.__cause__ is primary_cause
    assert cleanup_paths == staging_paths
    assert staging_paths[0].is_dir()
    assert not (cache / f"Liger-Kernel-{targets.LIGER_REVISION}").exists()


def test_fetch_reports_cleanup_failure_after_promotion(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cleanup_error = OSError("cleanup failed")
    cleanup_paths = []
    validated = []
    monkeypatch.setattr(
        target_support.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(_archive_payload()),
    )
    monkeypatch.setattr(liger_packet, "_validate_liger_tree", validated.append)

    def fail_cleanup(path):
        cleanup_paths.append(path)
        raise cleanup_error

    monkeypatch.setattr(target_support.shutil, "rmtree", fail_cleanup)
    with pytest.raises(
        TargetError,
        match="could not remove Liger cache staging directory",
    ) as raised:
        targets.fetch_liger(cache)

    root = cache / f"Liger-Kernel-{targets.LIGER_REVISION}"
    assert raised.value.__cause__ is cleanup_error
    assert cleanup_paths
    assert validated
    assert str(cleanup_paths[0]) in str(raised.value)
    assert root.is_dir()
    assert (root / "LICENSE").read_text(encoding="utf-8") == "license"


@pytest.mark.parametrize(
    "cleanup_fails",
    (False, True),
    ids=("cleanup-succeeds", "cleanup-fails"),
)
def test_fetch_reuses_cache_root_that_appears_after_download(
    monkeypatch, tmp_path, cleanup_fails
):
    cache = tmp_path / "cache"
    validated = []
    cleanup_error = OSError("cleanup failed")
    cleanup_paths = []
    monkeypatch.setattr(
        target_support.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(_archive_payload()),
    )

    def validate(root):
        validated.append(root)
        if len(validated) == 1:
            (cache / root.name).mkdir()

    monkeypatch.setattr(liger_packet, "_validate_liger_tree", validate)
    if cleanup_fails:

        def fail_cleanup(path):
            cleanup_paths.append(path)
            raise cleanup_error

        monkeypatch.setattr(target_support.shutil, "rmtree", fail_cleanup)

    if cleanup_fails:
        with pytest.raises(
            TargetError,
            match="could not remove Liger cache staging directory",
        ) as raised:
            targets.fetch_liger(cache)
        assert raised.value.__cause__ is cleanup_error
        assert cleanup_paths
        assert str(cleanup_paths[0]) in str(raised.value)
    else:
        result = targets.fetch_liger(cache)

    assert len(validated) == 2
    assert validated[0].parent.parent == cache
    winner = cache / f"Liger-Kernel-{targets.LIGER_REVISION}"
    assert validated[1] == winner
    assert validated[0] != winner
    assert winner.is_dir()
    if cleanup_fails:
        assert validated[0].parent.exists()
        assert cleanup_paths == [validated[0].parent]
    else:
        assert result == winner
        assert not validated[0].parent.exists()
        assert list(cache.iterdir()) == [result]


def test_fetch_validates_then_atomically_promotes(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    validated = []
    monkeypatch.setattr(
        target_support.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(_archive_payload()),
    )
    monkeypatch.setattr(liger_packet, "_validate_liger_tree", validated.append)

    root = targets.fetch_liger(cache)

    assert len(validated) == 1
    assert validated[0] != root
    assert validated[0].name == root.name
    assert validated[0].parent.parent == cache
    assert root == cache / f"Liger-Kernel-{targets.LIGER_REVISION}"
    assert (root / "LICENSE").read_text(encoding="utf-8") == "license"
    assert list(cache.iterdir()) == [root]


@pytest.mark.parametrize(
    ("name", "display_name", "failure"),
    (
        ("torch", "PyTorch", ModuleNotFoundError("torch")),
        ("triton", "Triton", ModuleNotFoundError("triton")),
        ("torch", "PyTorch", OSError("native loader failed")),
    ),
)
def test_unavailable_runtime_module_is_target_error(monkeypatch, name, display_name, failure):

    def unavailable(requested):
        if requested == name:
            raise failure
        raise AssertionError(requested)

    monkeypatch.setattr(target_support.importlib, "import_module", unavailable)
    with pytest.raises(TargetError) as raised:
        targets._require_module(name, display_name)
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize(
    ("target_name", "name", "display_name"),
    (
        ("Liger", "torch", "PyTorch"),
        ("Liger", "triton", "Triton"),
        ("ART", "torch", "PyTorch"),
        ("ART", "triton", "Triton"),
    ),
)
def test_gpu_environment_reports_explicit_target_and_preserves_module_cause(
    monkeypatch, target_name, name, display_name
):
    failure = ModuleNotFoundError(name)

    def unavailable(requested):
        if requested == name:
            raise failure
        return SimpleNamespace()

    monkeypatch.setattr(target_support.importlib, "import_module", unavailable)
    with pytest.raises(
        TargetError,
        match=rf"the {target_name} demonstration requires {display_name}",
    ) as raised:
        target_support._gpu_environment(target_name)

    assert raised.value.__cause__ is failure


def test_liger_inductor_import_normalizes_native_loader_oserror(monkeypatch):
    failure = OSError("native loader failed")
    real_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "torch":
            return SimpleNamespace()
        if name == "torch._inductor":
            raise failure
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(TargetError, match="PyTorch Inductor is unavailable") as raised:
        targets._compile_lane(None, None, None, None)

    assert raised.value.__cause__ is failure


def test_liger_load_validates_before_wrapping_import_failure(monkeypatch, tmp_path):
    events = []
    package_paths = []
    failure = ImportError("missing import")
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "liger_kernel" or name.startswith("liger_kernel.")
    }

    def validate(root):
        events.append(("validate", root))

    def unavailable(name, _path, _expected, _target_name):
        events.append(("load", name))
        package_paths.append(tuple(sys.modules["liger_kernel"].__path__))
        raise failure

    monkeypatch.setattr(liger_packet, "_validate_liger_tree", validate)
    monkeypatch.setattr(source_files, "_load_exact_module", unavailable)
    try:
        with pytest.raises(TargetError) as raised:
            targets._load_liger_function(tmp_path)
    finally:
        for name in tuple(sys.modules):
            if name == "liger_kernel" or name.startswith("liger_kernel."):
                del sys.modules[name]
        sys.modules.update(previous)
    assert events[:2] == [("validate", tmp_path), ("load", "liger_kernel.utils")]
    assert package_paths == [()]
    assert raised.value.__cause__ is failure


def test_exact_module_load_executes_source_with_import_metadata(tmp_path):
    path = tmp_path / "normal.py"
    path.write_text(
        "loaded = (__name__, __package__, __file__, __spec__.name)\n",
        encoding="utf-8",
    )
    name = "liger_kernel.ops.normal"

    try:
        module = source_files._load_exact_module(
            name, path, hashlib.sha256(path.read_bytes()).hexdigest()
        )
    finally:
        loaded = sys.modules.pop(name, None)

    assert module is loaded
    assert module.loaded == (name, "liger_kernel.ops", str(path), name)
    assert module.__loader__ is None
    assert module.__cached__ is None


def test_exact_module_load_removes_failed_module_after_path_validation(tmp_path):
    path = tmp_path / "wrong_path.py"
    path.write_text("__file__ = 'different.py'\n", encoding="utf-8")
    name = "liger_kernel.ops.wrong_path"

    with pytest.raises(TargetError, match="unexpected path"):
        source_files._load_exact_module(
            name, path, hashlib.sha256(path.read_bytes()).hexdigest()
        )

    assert name not in sys.modules


def test_exact_module_loader_declares_its_clean_process_imports(tmp_path):
    path = tmp_path / "isolated.py"
    path.write_text("value = 7\n", encoding="utf-8")
    source_root = Path(__file__).parents[1] / "src"
    script = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            f"sys.path.insert(0, {str(source_root)!r})",
            "from daggertrace.source_files import _load_exact_module",
            f"module = _load_exact_module('fixture.isolated', Path({str(path)!r}), "
            f"{hashlib.sha256(path.read_bytes()).hexdigest()!r})",
            "assert module.value == 7",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_exact_module_rejects_a_hash_mismatch_before_execution(tmp_path):
    path = tmp_path / "mismatch.py"
    path.write_text("raise AssertionError('source was executed')\n", encoding="utf-8")

    with pytest.raises(TargetError, match="hash mismatch"):
        source_files._load_exact_module("liger_kernel.ops.mismatch", path, "0" * 64)

    assert "liger_kernel.ops.mismatch" not in sys.modules


def test_exact_module_load_ignores_timestamp_valid_adjacent_bytecode(tmp_path):
    path = tmp_path / "cached.py"
    cached_source = "value = 'cached'\n"
    checked_source = "value = 'source'\n"
    assert len(cached_source) == len(checked_source)
    path.write_text(cached_source, encoding="utf-8")
    metadata = path.stat()
    py_compile.compile(
        str(path),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    path.write_text(checked_source, encoding="utf-8")
    os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    name = "liger_kernel.ops.cached"
    cached_spec = importlib.util.spec_from_file_location(name, path)
    assert cached_spec is not None and cached_spec.loader is not None
    cached_module = importlib.util.module_from_spec(cached_spec)
    cached_spec.loader.exec_module(cached_module)
    assert cached_module.value == "cached"

    try:
        module = source_files._load_exact_module(
            name, path, hashlib.sha256(path.read_bytes()).hexdigest()
        )
    finally:
        sys.modules.pop(name, None)

    assert module.value == "source"


def test_reproducer_rejects_a_missing_witness(monkeypatch, tmp_path):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        float32=object(),
        manual_seed=lambda _seed: None,
        randn=lambda *_args, **_kwargs: object(),
    )
    evidence = _valid_runtime_evidence()
    evidence["candidate_vs_reference"]["gx_max_abs"] = 1e-4
    monkeypatch.setattr(
        targets,
        "_require_module",
        lambda name, _display_name: fake_torch if name == "torch" else object(),
    )
    monkeypatch.setattr(targets, "_load_liger_function", lambda _root: object())
    monkeypatch.setattr(targets, "_reference_function", lambda: object())
    monkeypatch.setattr(targets, "_liger_function", lambda *_args: object())
    monkeypatch.setattr(targets, "_run_lane", lambda *_args: object())
    monkeypatch.setattr(targets, "_comparison", lambda *_args: evidence)
    monkeypatch.setattr(targets, "_require_exact_gpu_framework", lambda *_args: None)
    with pytest.raises(TargetError, match="did not produce the expected numerical witness"):
        run_liger_reproducer(tmp_path)


def test_reproducer_rejects_cpu_only_runtime_before_allocation(monkeypatch, tmp_path):
    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("CUDA allocation was attempted")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        float32=object(),
        manual_seed=lambda _seed: None,
        randn=unexpected_allocation,
    )
    monkeypatch.setattr(
        targets,
        "_require_module",
        lambda name, _display_name: fake_torch if name == "torch" else object(),
    )
    monkeypatch.setattr(targets, "_load_liger_function", lambda _root: object())
    with pytest.raises(TargetError, match="reproducer requires CUDA"):
        run_liger_reproducer(tmp_path)


@pytest.mark.parametrize(
    ("warmups", "samples"),
    [(True, 3), (0, 3), (11, 3), (1, True), (1, 2), (1, 51)],
)
def test_audit_rejects_unbounded_benchmark_before_runtime(monkeypatch, tmp_path, warmups, samples):
    monkeypatch.setattr(
        targets,
        "_require_module",
        lambda *_args: (_ for _ in ()).throw(AssertionError("runtime import reached")),
    )
    with pytest.raises(TargetError, match="benchmark (warmups|samples)"):
        audit_liger(tmp_path / "audit", warmups=warmups, samples=samples)


def test_generated_reproducer_uses_deterministic_json_main(monkeypatch, tmp_path, capsys):
    targets._write_reproducer(tmp_path)
    path = tmp_path / "reproducer.py"
    script = path.read_text(encoding="utf-8")
    compile(script, "reproducer.py", "exec")
    observed = []

    def reproduce(target):
        observed.append(target)
        return {"z": 1, "a": 2}

    monkeypatch.setattr(targets, "run_liger_reproducer", reproduce)
    previous = sys.dont_write_bytecode
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.dont_write_bytecode = previous
    assert observed == [tmp_path / "external" / "liger" / "tree"]
    assert json.loads(capsys.readouterr().out) == {"a": 2, "z": 1}
