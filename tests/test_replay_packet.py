from __future__ import annotations

import copy
import json
import os
import pickle
from pathlib import Path

import pytest

from daggertrace.certificate import write_certificate
from daggertrace.replay import (
    MAX_CERTIFICATE_BYTES,
    ReplayError,
    replay_certificate,
    replay_source_evidence,
)
from daggertrace.replay_packet import (
    _State,
    _benchmark_evidence,
    _bind_runtime_environment,
    _bounded_file,
    _checker_source,
    _integer,
    _packet_sidecars,
    _sha,
    _sha_bytes,
)
from daggertrace.replay_v2 import _analysis, _capture, _execute, _validate_role_domains
from daggertrace.validation_v2_fixtures import _micrograph
from _ownership_fixtures import (
    _insert_connected_clone_before_accumulation,
    _with_explicit_saved_output,
)
from _replay_fixtures import (
    _canonical_storage_root_case,
    _replayable_micrograph,
    _sync_hop_getitems,
    _write_canonical,
)
from _certificate_fixtures import _packet


_TEST_REPRODUCER = b"import json\nprint(json.dumps({'status': 'ok'}, sort_keys=True))\n"


def test_replay_facade_keeps_public_objects_at_the_established_import_path():
    for exported in (ReplayError, replay_certificate, replay_source_evidence):
        assert exported.__module__ == "daggertrace.replay"
        assert pickle.loads(pickle.dumps(exported)) is exported

    error = pickle.loads(pickle.dumps(ReplayError("replay failed")))
    assert type(error) is ReplayError
    assert str(error) == "replay failed"


def _write_closed_packet(path, certificate):
    source = b"@triton.jit\ndef kernel(x_ptr):\n    tl.store(x_ptr, 0.0)\n"
    source_path = path / "external" / "source" / "kernel.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source)
    certificate.setdefault(
        "target",
        {
            "repository": "https://example.invalid/repository",
            "revision": "a" * 40,
            "files": [
                {
                    "packet_path": "external/source/kernel.py",
                    "target_path": "src/kernel.py",
                    "size": len(source),
                    "sha256": _sha_bytes(source),
                    "role": "source",
                }
            ],
            "kernels": [
                {
                    "module": "synthetic.packet",
                    "qualname": "kernel",
                    "file": "external/source/kernel.py",
                    "source_sha256": _sha_bytes(source),
                }
            ],
            "specialization": {},
        },
    )
    _write_canonical(path / "certificate.json", certificate)
    filenames = {
        "candidate.capture.json": "candidate",
        "control.capture.json": "control",
        "runtime.json": "runtime",
        "benchmark.json": "benchmark",
        "controls.json": "controls",
    }
    for filename, section in filenames.items():
        _write_canonical(path / filename, certificate[section])
    (path / "reproducer.py").write_bytes(_TEST_REPRODUCER)


def test_replay_rejects_collapse_from_same_id_same_layout_allocations():
    candidate, control = _same_id_same_layout_allocation_attack()
    candidate_result = _execute(candidate)
    control_result = _execute(control)

    with pytest.raises(ReplayError, match="exact candidate/control contribution substitution"):
        _analysis(candidate, control, candidate_result, control_result)


def test_replay_resealed_paired_allocations_do_not_share_a_coarse_contribution_part():
    candidate, control = _paired_allocation_outputs()
    candidate_result = _execute(candidate)
    control_result = _execute(control)

    with pytest.raises(ReplayError, match="marked gradient-accumulation add"):
        _analysis(candidate, control, candidate_result, control_result)


def test_replay_matches_paired_hops_across_capture_local_node_insertions():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    candidate = _replayable_micrograph(candidate)
    control = _replayable_micrograph(control)
    candidate_raw, control_raw = candidate, control
    candidate_result = _execute(candidate_raw)
    control_result = _execute(control_raw)

    derived = _analysis(candidate_raw, control_raw, candidate_result, control_result)

    assert derived["outcome"] == "VIOLATION"
    assert derived["reasons"] == ["BORROWED_COTANGENT_WRITE", "CONTRIBUTION_COLLAPSE"]
    assert derived["collapse"]["replaced_part"] == "tangent:output:0"


def test_public_replay_roundtrips_an_explicit_saved_output_role(tmp_path):
    path, certificate = _packet(tmp_path)
    candidate = _with_explicit_saved_output(certificate["candidate"])
    candidate = _capture(candidate, "candidate")
    candidate_result = _execute(candidate)

    certificate["candidate"] = candidate
    certificate["analysis"] = _analysis(candidate, None, candidate_result, None)
    write_certificate(path, certificate, reproducer_bytes=_TEST_REPRODUCER)

    assert replay_certificate(path)["outcome"] == "SAFE_IN_FRAGMENT"


def test_replay_rejects_a_connected_clone_outside_the_allowed_substitution():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    candidate = candidate.to_dict()
    control = control.to_dict()
    _insert_connected_clone_before_accumulation(control)
    candidate_result = _execute(candidate)
    control_result = _execute(control)

    with pytest.raises(ReplayError, match="normalized backward operator skeletons differ"):
        _analysis(candidate, control, candidate_result, control_result)


def test_replay_rejects_borrowed_paired_control_but_keeps_canonical_and_candidate_only():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    candidate_raw = candidate.to_dict()
    control_raw = control.to_dict()
    candidate_result = _execute(candidate_raw)

    canonical = _analysis(
        candidate_raw,
        control_raw,
        candidate_result,
        _execute(control_raw),
    )
    assert canonical["outcome"] == "VIOLATION"
    assert canonical["reasons"] == ["BORROWED_COTANGENT_WRITE", "CONTRIBUTION_COLLAPSE"]
    assert canonical["collapse"] is not None

    malicious_control = _redirect_hop_write_to_tangent(control)
    malicious_result = _execute(malicious_control)
    assert malicious_result["borrowed"]

    with pytest.raises(ReplayError, match="TWIN_RELATION: control graph contains borrowed writes"):
        _analysis(candidate_raw, malicious_control, candidate_result, malicious_result)

    candidate_only = _analysis(candidate_raw, None, candidate_result, None)
    assert candidate_only["outcome"] == "VIOLATION"
    assert candidate_only["reasons"] == ["BORROWED_COTANGENT_WRITE"]
    assert candidate_only["collapse"] is None
    assert candidate_only["control_digest"] is None


def test_replay_keeps_distinct_hop_occurrences_exact():
    candidate, control = _micrograph(True, "direct", False, 1, False, hops=2)
    candidate = _replayable_micrograph(candidate)
    control = _replayable_micrograph(control)
    second_hop = [node for node in control["backward"]["nodes"] if node["target"] == "triton.functional"][1]
    second_hop["kwargs"]["grid"] = [{"tuple": [2, 1, 1]}]
    candidate_result = _execute(candidate)
    control_result = _execute(control)

    with pytest.raises(ReplayError, match="exact candidate/control contribution substitution"):
        _analysis(candidate, control, candidate_result, control_result)


def test_replay_rejects_duplicate_tangent_output_roles_and_accepts_the_canonical_role():
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    assert len(_execute(candidate.to_dict())["borrowed"]) == 1

    duplicate = candidate.to_dict()
    view = next(node for node in duplicate["backward"]["nodes"] if node["target"] == "aten.view")
    view.update(
        op="placeholder",
        target="placeholder",
        args={"tuple": []},
        kwargs={},
        role=copy.deepcopy(duplicate["backward"]["nodes"][0]["role"]),
    )
    view["tensor"]["alias_group"] = "duplicate-tangent"

    with pytest.raises(ReplayError, match="duplicate tangent output role tangent:output:0"):
        _execute(duplicate)


def test_replay_rejects_a_backward_graph_without_a_tangent_placeholder():
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    raw = candidate.to_dict()
    raw["backward"]["nodes"][0]["role"] = None

    with pytest.raises(ReplayError, match="no tangent placeholders"):
        _validate_role_domains(raw, "candidate", {0}, {0})


def test_capture_requires_exactly_one_forward_and_backward_callback():
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    raw = copy.deepcopy(_replayable_micrograph(candidate))
    raw["diagnostics"]["callbacks"]["forward"] = 2

    with pytest.raises(ReplayError, match="callbacks.forward must equal one"):
        _capture(raw, "candidate")


def test_capture_requires_a_gradient_for_every_differentiable_input():
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    raw = candidate.to_dict()

    with pytest.raises(ReplayError, match="do not match differentiable inputs"):
        _validate_role_domains(raw, "candidate", {0}, {0, 1})


def test_capture_requires_a_primal_role_for_every_runtime_input():
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    raw = copy.deepcopy(_replayable_micrograph(candidate))
    extra = copy.deepcopy(raw["diagnostics"]["inputs"][0])
    extra["requires_grad"] = False
    raw["diagnostics"]["inputs"].append(extra)

    with pytest.raises(ReplayError, match="do not match runtime inputs"):
        _capture(raw, "candidate")


def test_runtime_environment_is_bound_to_capture_framework_identity():
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    capture = _replayable_micrograph(candidate)
    environment = {
        key: capture["framework"][key]
        for key in ("python", "torch", "torch_git", "triton", "cuda_runtime")
    }
    _bind_runtime_environment(environment, capture, None)
    environment["python"] = "different"

    with pytest.raises(ReplayError, match="differs from the candidate capture"):
        _bind_runtime_environment(environment, capture, None)


def _paired_allocation_outputs():
    candidate, control = _micrograph(False, "none", False, 1, False)

    def reseal(source, shape, alias_group):
        raw = source.to_dict()
        allocation = next(
            node
            for node in raw["backward"]["nodes"]
            if node["target"] == "aten.zeros_like"
        )
        allocation["target"] = "aten.empty"
        allocation["args"] = {"tuple": [list(shape)]}
        allocation["kwargs"] = {
            "dtype": {"torch": "float32"},
            "device": "accelerator",
        }
        allocation["tensor"].update(
            shape=list(shape),
            stride=[shape[1], 1],
            alias_group=alias_group,
        )
        _sync_hop_getitems(raw)
        raw["backward"]["nodes"][-1]["args"] = {
            "tuple": [{"tuple": [{"ref": allocation["id"]}]}]
        }
        return raw

    return reseal(candidate, (2, 2), "candidate-empty"), reseal(
        control, (3, 3), "control-empty"
    )


def _same_id_same_layout_allocation_attack():
    candidate, control = _micrograph(True, "direct", False, 1, True)

    def redirect(capture):
        raw = copy.deepcopy(capture.to_dict())
        allocation = next(node for node in raw["backward"]["nodes"] if node["id"] == "n2")
        assert allocation["target"] == "aten.empty"
        assert allocation["args"] == {"tuple": [[1]]}
        accumulation = next(
            node for node in raw["backward"]["nodes"] if node["target"] == "aten.add"
        )
        accumulation["args"]["tuple"][1] = {"ref": "n2"}
        return raw

    return redirect(candidate), redirect(control)


@pytest.mark.parametrize(
    "target",
    [
        "aten.zeros_like",
        "aten.empty",
        "aten.clone",
        "aten.mul",
        "aten.sum",
        "aten.add",
        "operator.getitem",
    ],
)
def test_replay_accepts_canonical_storage_root_opposites(target):
    capture, root = _canonical_storage_root_case(target)

    result = _execute(capture)

    assert result["borrowed"] == []
    assert root["tensor"]["alias_group"] not in {None, "tangent"}


@pytest.mark.parametrize(
    "target",
    [
        "aten.zeros_like",
        "aten.empty",
        "aten.clone",
        "aten.mul",
        "aten.sum",
        "aten.add",
        "operator.getitem",
    ],
)
@pytest.mark.parametrize("alias_group", [None, "tangent"])
def test_replay_rejects_resealed_storage_root_collision(target, alias_group):
    capture, root = _canonical_storage_root_case(target)
    root["tensor"]["alias_group"] = alias_group

    with pytest.raises(ReplayError, match="captured storage identity|fresh result|result metadata"):
        _execute(capture)


@pytest.mark.parametrize("target", ["aten.view", "aten._unsafe_view", "aten.alias", "aten.detach"])
def test_replay_accepts_canonical_view_storage_identity(target):
    candidate, _ = _micrograph(True, "none", False, 1, False)
    capture = candidate.to_dict()
    view = next(node for node in capture["backward"]["nodes"] if node["target"] == "aten.view")
    view["target"] = target

    result = _execute(capture)

    assert len(result["borrowed"]) == 1


@pytest.mark.parametrize("target", ["aten.view", "aten._unsafe_view", "aten.alias", "aten.detach"])
def test_replay_rejects_resealed_view_storage_mismatch(target):
    _, control = _micrograph(True, "none", False, 1, False)
    capture = control.to_dict()
    view = next(node for node in capture["backward"]["nodes"] if node["target"] == "aten.view")
    view["target"] = target
    view["tensor"]["alias_group"] = "resealed-view"

    with pytest.raises(ReplayError, match="differs from input"):
        _execute(capture)


def _forward_empty_storage_cases(alias_group):
    candidate, control = _micrograph(False, "none", False, 0, False)
    captures = []
    for source in (candidate, control):
        capture = source.to_dict()
        nodes = capture["forward"]["nodes"]
        allocation = copy.deepcopy(
            next(node for node in capture["backward"]["nodes"] if node["target"] == "aten.zeros_like")
        )
        allocation["id"] = "n1"
        allocation["target"] = "aten.empty"
        allocation["args"] = {"tuple": [[2, 2]]}
        allocation["kwargs"] = {
            "dtype": {"torch": "float32"},
            "device": "accelerator",
        }
        allocation["tensor"]["alias_group"] = alias_group
        output = nodes[-1]
        output["id"] = "n2"
        output["args"] = {"tuple": [{"tuple": [{"ref": "n1"}]}]}
        nodes.insert(1, allocation)
        captures.append(capture)
    return captures


def test_replay_accepts_canonical_forward_allocation_for_candidate_and_control():
    for capture in _forward_empty_storage_cases("forward-fresh"):
        assert _execute(capture)["borrowed"] == []


@pytest.mark.parametrize("alias_group", [None, "primal"])
def test_replay_rejects_forward_allocation_storage_collision(alias_group):
    for capture in _forward_empty_storage_cases(alias_group):
        with pytest.raises(ReplayError, match="fresh result"):
            _execute(capture)


def test_replay_rejects_open_or_unhashed_artifact(tmp_path):
    path = tmp_path / "certificate.json"
    _write_canonical(path, {"schema": "daggertrace.cotangent-ownership/1"})
    with pytest.raises(ReplayError, match="fields changed"):
        replay_certificate(path)


def test_replay_rejects_retired_certificate_schema(tmp_path):
    path, certificate = _packet(tmp_path)
    certificate["schema"] = "daggertrace.aot-storage-ownership/3"
    certificate["digest"] = _sha(
        {key: value for key, value in certificate.items() if key != "digest"}
    )
    _write_canonical(path, certificate)

    with pytest.raises(ReplayError, match="unsupported certificate schema"):
        replay_certificate(path)


@pytest.mark.parametrize(
    "encoded",
    [
        b'{"schema": "daggertrace.cotangent-ownership/1"}\n',
        b'{"schema":"daggertrace.cotangent-ownership/1","schema":"daggertrace.cotangent-ownership/1"}\n',
    ],
)
def test_replay_rejects_noncanonical_or_duplicate_key_bytes(tmp_path, encoded):
    path = tmp_path / "certificate.json"
    path.write_bytes(encoded)
    with pytest.raises(ReplayError, match="not canonical JSON"):
        replay_certificate(path)


def test_replay_wraps_missing_and_malformed_public_inputs(tmp_path):
    with pytest.raises(ReplayError, match="absent or unreadable"):
        replay_certificate(tmp_path / "missing.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_bytes(b"{\n")
    with pytest.raises(ReplayError, match="could not be replayed"):
        replay_certificate(malformed)


@pytest.mark.skipif(os.name != "posix", reason="symlink-loop resolution is POSIX coverage")
def test_replay_wraps_symlink_loop_resolution_for_both_public_routes(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.symlink_to(second, target_is_directory=True)
    second.symlink_to(first, target_is_directory=True)

    with pytest.raises(ReplayError):
        replay_certificate(first / "certificate.json")
    with pytest.raises(ReplayError):
        replay_source_evidence(first / "source-evidence.json")


def test_replay_rejects_oversized_certificate_before_parsing(tmp_path):
    path = tmp_path / "oversized.json"
    with path.open("wb") as handle:
        handle.seek(MAX_CERTIFICATE_BYTES)
        handle.write(b"x")
    with pytest.raises(ReplayError, match="byte limit"):
        replay_certificate(path)


def test_replay_rejects_primary_symlink_before_open(tmp_path):
    target = tmp_path / "target.json"
    _write_canonical(target, {"schema": "daggertrace.cotangent-ownership/1"})
    path = tmp_path / "certificate.json"
    path.symlink_to(target)
    with pytest.raises(ReplayError, match="traverses a symlink"):
        replay_certificate(path)


def test_replay_rejects_primary_fifo_before_open(tmp_path):
    path = tmp_path / "certificate.json"
    os.mkfifo(path)
    with pytest.raises(ReplayError, match="not a regular file"):
        replay_certificate(path)


def test_replay_wraps_huge_json_integer(tmp_path):
    path = tmp_path / "huge-integer.json"
    path.write_text('{"value":' + "9" * 5_000 + "}\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="could not be replayed"):
        replay_certificate(path)


def test_replay_rejects_excessive_canonical_nesting(tmp_path):
    value = []
    for _ in range(66):
        value = [value]
    path = tmp_path / "deep.json"
    _write_canonical(path, value)
    with pytest.raises(ReplayError, match="nesting limit"):
        replay_certificate(path)


@pytest.mark.parametrize("value", [True, -1])
def test_replay_integer_slots_reject_bool_and_negative(value):
    with pytest.raises(ReplayError):
        _integer(value, "slot", minimum=0)


def test_packet_sidecars_exactly_match_embedded_sections(tmp_path):
    certificate = {
        "candidate": {"lane": "candidate"},
        "control": {"lane": "control"},
        "runtime": {"evidence": "runtime"},
        "benchmark": {"evidence": "benchmark"},
        "controls": {"evidence": "controls"},
    }
    _write_closed_packet(tmp_path, certificate)

    certificate_path = tmp_path / "certificate.json"
    _packet_sidecars(certificate, certificate_path)

    (tmp_path / "runtime.json").write_bytes(
        (tmp_path / "runtime.json").read_bytes() + b" "
    )
    with pytest.raises(ReplayError, match="runtime.json differs"):
        _packet_sidecars(certificate, certificate_path)


def test_packet_sidecars_accept_target_owned_reproducer_bytes(tmp_path):
    certificate = {
        "candidate": {"lane": "candidate"},
        "control": {"lane": "control"},
        "runtime": {"evidence": "runtime"},
        "benchmark": {"evidence": "benchmark"},
        "controls": {"evidence": "controls"},
    }
    _write_closed_packet(tmp_path, certificate)
    (tmp_path / "reproducer.py").write_bytes(
        b"import json\nprint(json.dumps({'target': 'owned'}, sort_keys=True))\n"
    )

    _packet_sidecars(certificate, tmp_path / "certificate.json")


def test_packet_inventory_rejects_a_renamed_complete_certificate(tmp_path):
    certificate = {
        "candidate": {},
        "control": {},
        "runtime": {},
        "benchmark": {},
        "controls": {},
    }
    _write_closed_packet(tmp_path, certificate)
    renamed = tmp_path / "renamed.json"
    (tmp_path / "certificate.json").rename(renamed)

    with pytest.raises(ReplayError, match="packet certificate filename changed"):
        _packet_sidecars(certificate, renamed)


@pytest.mark.parametrize("kind", ["regular", "symlink", "fifo", "directory"])
def test_packet_inventory_rejects_every_unexpected_entry_kind(tmp_path, kind):
    certificate = {
        "candidate": {},
        "control": {},
        "runtime": {},
        "benchmark": {},
        "controls": {},
    }
    _write_closed_packet(tmp_path, certificate)
    unexpected = tmp_path / f"unexpected-{kind}"
    if kind == "regular":
        unexpected.write_bytes(b"untrusted")
    elif kind == "symlink":
        unexpected.symlink_to(tmp_path / "reproducer.py")
    elif kind == "fifo":
        os.mkfifo(unexpected)
    else:
        unexpected.mkdir()
        os.mkfifo(unexpected / "must-not-be-traversed")

    with pytest.raises(ReplayError, match="packet inventory contains unexpected entry"):
        _packet_sidecars(certificate, tmp_path / "certificate.json")


def test_packet_inventory_rejects_nested_bytecode_hitchhiker(tmp_path):
    certificate = {
        "candidate": {},
        "control": {},
        "runtime": {},
        "benchmark": {},
        "controls": {},
    }
    _write_closed_packet(tmp_path, certificate)
    bytecode = tmp_path / "external" / "source" / "__pycache__" / "kernel.cpython-314.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"untrusted bytecode")

    with pytest.raises(
        ReplayError,
        match="unexpected entry external/source/__pycache__",
    ):
        _packet_sidecars(certificate, tmp_path / "certificate.json")


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_packet_inventory_rejects_nonregular_required_leaf(tmp_path, kind):
    certificate = {
        "candidate": {},
        "control": {},
        "runtime": {},
        "benchmark": {},
        "controls": {},
    }
    _write_closed_packet(tmp_path, certificate)
    source_path = tmp_path / "external" / "source" / "kernel.py"
    source_path.unlink()
    if kind == "symlink":
        source_path.symlink_to(tmp_path / "reproducer.py")
    else:
        os.mkfifo(source_path)

    with pytest.raises(ReplayError, match="is not a regular file"):
        _packet_sidecars(certificate, tmp_path / "certificate.json")


def test_packet_inventory_rejects_missing_required_leaf(tmp_path):
    certificate = {
        "candidate": {},
        "control": {},
        "runtime": {},
        "benchmark": {},
        "controls": {},
    }
    _write_closed_packet(tmp_path, certificate)
    missing = tmp_path / "external" / "source" / "kernel.py"
    missing.unlink()

    with pytest.raises(ReplayError, match="is missing kernel.py"):
        _packet_sidecars(certificate, tmp_path / "certificate.json")


def test_external_claimed_leaf_cannot_symlink_to_host_file(tmp_path):
    packet = tmp_path / "packet"
    claimed = packet / "external" / "source" / "kernel.py"
    claimed.parent.mkdir(parents=True)
    host_file = tmp_path / "host-license"
    host_file.write_bytes(b"host bytes")
    claimed.symlink_to(host_file)
    with pytest.raises(ReplayError, match="traverses a symlink"):
        _bounded_file(claimed, 100, "external license", root=packet)


def test_external_claimed_path_cannot_traverse_a_symlinked_directory(tmp_path):
    packet = tmp_path / "packet"
    packet.mkdir()
    host = tmp_path / "host"
    host.mkdir()
    (host / "kernel.py").write_bytes(b"host bytes")
    (packet / "external").symlink_to(host)

    with pytest.raises(ReplayError, match="traverses a symlink"):
        _bounded_file(
            packet / "external" / "kernel.py",
            100,
            "external source",
            root=packet,
        )


def test_collapse_replacement_must_belong_to_the_same_tangent():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    candidate = candidate.to_dict()
    control = control.to_dict()
    candidate_producer = next(
        node["id"]
        for node in candidate["backward"]["nodes"]
        if node["target"] == "aten.add" and node["gradient_accumulation"]
    )
    control_producer = next(
        node["id"]
        for node in control["backward"]["nodes"]
        if node["target"] == "aten.add" and node["gradient_accumulation"]
    )
    candidate_result = {
        "gradients": {
            0: _State(
                "candidate", 0, ("hop:wrong", "stable"), None, candidate_producer
            )
        },
        "borrowed": [
            {
                "tangent": "tangent:output:1",
                "hop_node": "n4",
                "write_role": "dX_ptr",
            }
        ],
        "borrowed_parts": {("n4", "dX_ptr"): "hop:wrong"},
    }
    control_result = {
        "gradients": {
            0: _State(
                "control",
                0,
                ("tangent:output:0", "stable"),
                None,
                control_producer,
            )
        },
        "borrowed": [],
    }
    with pytest.raises(ReplayError, match="not an exact candidate/control contribution substitution"):
        _analysis(candidate, control, candidate_result, control_result)


def test_replay_rejects_duplicate_replaced_tangent_contributions():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    candidate = candidate.to_dict()
    control = control.to_dict()
    candidate_add = next(node for node in candidate["backward"]["nodes"] if node["target"] == "aten.add")
    control_add = next(node for node in control["backward"]["nodes"] if node["target"] == "aten.add")
    candidate_tangent = next(
        node
        for node in candidate["backward"]["nodes"]
        if node["role"] and node["role"].get("kind") == "tangent"
    )
    control_tangent = next(
        node
        for node in control["backward"]["nodes"]
        if node["role"] and node["role"].get("kind") == "tangent"
    )
    candidate_result = next(
        node for node in candidate["backward"]["nodes"] if node["target"] == "operator.getitem"
    )
    candidate_add["args"] = {
        "tuple": [{"ref": candidate_tangent["id"]}, {"ref": candidate_result["id"]}]
    }
    control_add["args"] = {
        "tuple": [{"ref": control_tangent["id"]}, {"ref": control_tangent["id"]}]
    }
    candidate_result = _execute(candidate)
    control_result = _execute(control)

    with pytest.raises(
        ReplayError,
        match="TWIN_RELATION: control gradient slot 0 repeats tangent:output:0",
    ):
        _analysis(candidate, control, candidate_result, control_result)


def _redirect_hop_write_to_tangent(capture):
    raw = copy.deepcopy(capture.to_dict())
    tangent = next(
        node
        for node in raw["backward"]["nodes"]
        if node["role"] and node["role"].get("kind") == "tangent"
    )
    hop = next(node for node in raw["backward"]["nodes"] if node["target"] == "triton.functional")
    write_role = hop["kwargs"]["writes"][0]
    hop["kwargs"]["bindings"][write_role] = {"ref": tangent["id"]}
    _sync_hop_getitems(raw)
    return raw


@pytest.mark.parametrize("warmups,samples", [(11, 3), (1, 51)])
def test_benchmark_evidence_rejects_values_above_producer_bounds(warmups, samples):
    benchmark = {
        "warmups": warmups,
        "samples": samples,
        "lanes": {},
        "claim": "supporting evidence only",
    }
    with pytest.raises(ReplayError, match="exceeds its maximum"):
        _benchmark_evidence(benchmark)


def test_replay_bounds_derived_gradient_contribution_growth():
    candidate, _control = _micrograph(False, "none", False, 1, False)
    raw = copy.deepcopy(candidate.to_dict())
    nodes = raw["backward"]["nodes"]
    output = nodes.pop()
    current = output["args"]["tuple"][0]["tuple"][0]["ref"]
    template = next(node for node in nodes if node["id"] == current)
    for _ in range(13):
        node_id = f"n{len(nodes)}"
        accumulation = copy.deepcopy(template)
        accumulation.update(
            id=node_id,
            op="call_function",
            target="aten.add",
            args={"tuple": [{"ref": current}, {"ref": current}]},
            kwargs={},
            role=None,
            gradient_accumulation=True,
        )
        accumulation["tensor"]["alias_group"] = f"derived-{node_id}"
        nodes.append(accumulation)
        current = node_id
        template = accumulation
    output["id"] = f"n{len(nodes)}"
    output["args"] = {"tuple": [{"tuple": [{"ref": current}]}]}
    nodes.append(output)

    with pytest.raises(ReplayError, match="derived-contribution work limit"):
        _execute(raw)


def test_checked_bundle_checker_identity_is_explicitly_replay_compatible():
    root = Path(__file__).resolve().parents[1]
    certificate = json.loads(
        (
            root
            / "evidence/results/bundle/liger/liger-rmsnorm/certificate.json"
        ).read_text(encoding="utf-8")
    )
    source = certificate["checker"]["source"]

    _checker_source(source)
    changed = copy.deepcopy(source)
    changed["production_loc"] += 1
    with pytest.raises(ReplayError, match="checker source identity is unsupported"):
        _checker_source(changed)
