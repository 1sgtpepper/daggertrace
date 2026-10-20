"""Certificate mutation fixtures and isolated packet replay helpers."""

from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from .canonical import sha256_bytes, sha256_json, write_canonical_json
from .model import Capture, Outcome, Reason
from .ownership import analyze
from .replay import ReplayError, replay_certificate
from .validation_v2 import V2_MUTATION_COUNT
from .validation_v2_fixtures import (
    _alternate_tensor_reference,
    _backward_kernel,
    _fresh_backward_allocation,
    _getitem,
    _gradient_add,
    _hop,
    _kernel_identity,
    _tangent_node,
)


def validate_certificate_mutations(
    certificate: dict[str, Any],
    certificate_path: Path,
    *,
    replay_fn=None,
) -> int:
    replay = replay_certificate if replay_fn is None else replay_fn
    mutations = _certificate_mutations(certificate)
    if len(mutations) != V2_MUTATION_COUNT:
        raise AssertionError(f"mutation denominator changed: {len(mutations)}")
    killed = 0
    for index, (name, mutant) in enumerate(mutations):
        _reseal_mutant(mutant)
        with tempfile.TemporaryDirectory(
            prefix=f"daggertrace-mutation-{index:02d}-{name}-"
        ) as temporary:
            path = _write_mutant_packet(
                mutant,
                packet=Path(temporary),
                source_packet=certificate_path.parent,
            )
            try:
                replay(path)
            except ReplayError:
                killed += 1
            else:
                raise AssertionError(f"certificate mutation survived: {name}")
    return killed


def _write_mutant_packet(
    certificate: dict[str, Any],
    *,
    packet: Path,
    source_packet: Path,
) -> Path:
    body = {key: value for key, value in certificate.items() if key != "digest"}
    if certificate.get("digest") != sha256_json(body):
        raise AssertionError("mutant certificate was not transitively resealed")
    shutil.copytree(source_packet / "external", packet / "external")
    shutil.copyfile(source_packet / "reproducer.py", packet / "reproducer.py")
    for filename, section in (
        ("candidate.capture.json", "candidate"),
        ("control.capture.json", "control"),
        ("runtime.json", "runtime"),
        ("benchmark.json", "benchmark"),
        ("controls.json", "controls"),
    ):
        write_canonical_json(packet / filename, certificate[section])
    path = packet / "certificate.json"
    write_canonical_json(path, certificate)
    return path


def _reseal_mutant(certificate: dict[str, Any]) -> None:
    body = {key: value for key, value in certificate.items() if key != "digest"}
    certificate["digest"] = sha256_json(body)


def _certificate_mutations(
    certificate: dict[str, Any], *, analyze_fn: Callable = analyze
) -> list[tuple[str, dict[str, Any]]]:
    return [
        *_corruption_mutations(certificate, analyze_fn=analyze_fn),
        *_closure_mutations(certificate, analyze_fn=analyze_fn),
        *_evidence_integrity_mutations(certificate),
    ]


def _corruption_mutations(
    certificate: dict[str, Any], *, analyze_fn: Callable = analyze
) -> list[tuple[str, dict[str, Any]]]:
    edits = (
        (
            "corrupt-kernel-source-hash",
            lambda c: _backward_kernel(c["candidate"]).update(
                source_sha256=_digest_token("corrupt-source-hash")
            ),
        ),
        (
            "corrupt-kernel-source",
            lambda c: _backward_kernel(c["candidate"]).update(
                source=_backward_kernel(c["candidate"])["source"] + "\n"
            ),
        ),
        (
            "corrupt-kernel-id",
            lambda c: _backward_kernel(c["candidate"]).update(
                id=_digest_token("corrupt-kernel-id")
            ),
        ),
        (
            "corrupt-kernel-read",
            lambda c: _backward_kernel(c["candidate"])["reads"].remove(
                _backward_kernel(c["candidate"])["reads"][0]
            ),
        ),
        (
            "corrupt-kernel-write",
            lambda c: _backward_kernel(c["candidate"])["writes"].remove(
                _backward_kernel(c["candidate"])["writes"][0]
            ),
        ),
        (
            "corrupt-source-read",
            lambda c: _backward_kernel(c["candidate"])["source_reads"].append(
                _backward_kernel(c["candidate"])["writes"][0]
            ),
        ),
        (
            "corrupt-source-write",
            lambda c: _backward_kernel(c["candidate"])["source_writes"].remove(
                _backward_kernel(c["candidate"])["writes"][0]
            ),
        ),
        (
            "corrupt-hop-kernel",
            lambda c: _hop(c["candidate"])["kwargs"].update(
                kernel=_digest_token("corrupt-hop-kernel")
            ),
        ),
        ("corrupt-hop-binding", lambda c: _corrupt_hop_binding(c["candidate"])),
        (
            "corrupt-hop-writes",
            lambda c: _hop(c["candidate"])["kwargs"]["writes"].remove(
                _hop(c["candidate"])["kwargs"]["writes"][0]
            ),
        ),
        (
            "corrupt-getitem-key",
            lambda c: _getitem(c["candidate"])["args"]["tuple"].__setitem__(
                1, "missing-output-role"
            ),
        ),
        (
            "corrupt-tangent-role",
            lambda c: _tangent_node(c["candidate"]).update(
                role={"kind": "saved", "descriptor": "x"}
            ),
        ),
        (
            "corrupt-output-slot",
            lambda c: c["candidate"]["backward"]["output_roles"][0]["primal"].update(
                index=c["candidate"]["backward"]["output_roles"][0]["primal"]["index"] + 1
            ),
        ),
        (
            "corrupt-candidate-add",
            lambda c: _gradient_add(c["candidate"])["args"]["tuple"].__setitem__(
                0, _alternate_tensor_reference(c["candidate"], _gradient_add(c["candidate"]))
            ),
        ),
        (
            "corrupt-control-add",
            lambda c: _gradient_add(c["control"])["args"]["tuple"].__setitem__(
                0, _gradient_add(c["control"])["args"]["tuple"][1]
            ),
        ),
        (
            "corrupt-unsupported-target",
            lambda c: _gradient_add(c["candidate"]).update(target="unsupported:unknown"),
        ),
        ("corrupt-capture-config", lambda c: c["candidate"]["capture_config"].update(dynamic=True)),
        (
            "corrupt-framework",
            lambda c: c["candidate"]["framework"].update(
                torch_git=_digest_token("corrupt-framework")[:40]
            ),
        ),
        (
            "corrupt-checker-source-file",
            lambda c: c["checker"]["source"]["files"].update(
                {next(iter(c["checker"]["source"]["files"])): _digest_token("corrupt-checker-file")}
            ),
        ),
        (
            "corrupt-checker-source-digest",
            lambda c: c["checker"]["source"].update(digest=_digest_token("corrupt-checker-digest")),
        ),
        ("corrupt-analysis-outcome", lambda c: c["analysis"].update(outcome="SAFE_IN_FRAGMENT")),
        (
            "corrupt-control-fresh-alias-group",
            lambda c: _collide_control_fresh_alias_group(c, analyze_fn=analyze_fn),
        ),
    )
    mutations = []
    for name, edit in edits:
        value = copy.deepcopy(certificate)
        edit(value)
        mutations.append((name, value))
    return mutations


def _closure_mutations(
    certificate: dict[str, Any], *, analyze_fn: Callable = analyze
) -> list[tuple[str, dict[str, Any]]]:
    edits = (
        (
            "closure-target-revision",
            lambda c: c["target"].update(revision=_digest_token("mutated-revision")[:40]),
            None,
        ),
        (
            "closure-target-source-snapshot",
            lambda c: _update_target_digest(
                c["target"], "source", _digest_token("mutated-source-snapshot")
            ),
            None,
        ),
        (
            "closure-target-specialization",
            lambda c: c["target"]["specialization"].update(expression="operation(y)"),
            None,
        ),
        (
            "closure-target-dependency",
            lambda c: _update_target_digest(
                c["target"], "dependency", _digest_token("mutated-dependency")
            ),
            None,
        ),
        ("closure-detached-kernel-relabel", _relabel_detached_kernel, None),
        (
            "closure-producer-rejected-source-effect",
            _inject_unsupported_source_effect,
            _refresh_rejected_source_analysis,
        ),
        ("closure-both-twin-unsupported-forward", _make_both_twins_unsupported, None),
    )
    mutations = []
    for name, edit, refresh in edits:
        value = copy.deepcopy(certificate)
        edit(value)
        if refresh is None:
            _refresh_mutation_closure(value, analyze_fn=analyze_fn)
        else:
            refresh(value)
        mutations.append((name, value))
    return mutations


def _evidence_integrity_mutations(
    certificate: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    edits = (
        ("integrity-runtime-single-field-contradiction", _zero_runtime_cold_time),
        (
            "integrity-benchmark-sample-count",
            lambda c: next(iter(c["benchmark"]["lanes"].values()))["device_ms"].pop(),
        ),
        ("integrity-runtime-no-witness", _erase_runtime_witness),
        ("integrity-benchmark-nonpositive", _make_benchmark_nonpositive),
        ("integrity-validation-fabricated-counts", _fabricate_validation_counts),
    )
    mutations = []
    for name, edit in edits:
        value = copy.deepcopy(certificate)
        edit(value)
        mutations.append((name, value))
    return mutations


def _relabel_detached_kernel(certificate: dict[str, Any]) -> None:
    _rewrite_twin_backward_kernel(
        certificate,
        lambda kernel: kernel.update(module="renamed.module"),
    )


def _inject_unsupported_source_effect(certificate: dict[str, Any]) -> None:
    _rewrite_twin_backward_kernel(certificate, _append_unsupported_source_effect)


def _append_unsupported_source_effect(kernel: dict[str, Any]) -> None:
    write_role = kernel["writes"][0]
    kernel["source"] = f"{kernel['source'].rstrip()}\n    tl.atomic_cas({write_role}, 0, 0)\n"


def _make_both_twins_unsupported(certificate: dict[str, Any]) -> None:
    for capture in (certificate["candidate"], certificate["control"]):
        output = next(node for node in capture["forward"]["nodes"] if node["op"] == "output")
        output["target"] = "unsupported:output"


def _collide_control_fresh_alias_group(
    certificate: dict[str, Any], *, analyze_fn: Callable = analyze
) -> None:
    allocation = _fresh_backward_allocation(certificate["control"])
    tangent = _tangent_node(certificate["control"])
    allocation_tensor = allocation["tensor"]
    tangent_tensor = tangent["tensor"]
    if allocation_tensor is None or tangent_tensor is None:
        raise AssertionError("control freshness mutation requires tensor metadata")
    tangent_group = tangent_tensor["alias_group"]
    if not isinstance(tangent_group, str) or not tangent_group:
        raise AssertionError("control freshness mutation requires a tangent storage identity")
    allocation_tensor["alias_group"] = tangent_group
    _getitem(certificate["control"])["tensor"] = copy.deepcopy(allocation_tensor)
    _refresh_mutation_closure(certificate, analyze_fn=analyze_fn)


def _rewrite_twin_backward_kernel(certificate: dict[str, Any], edit) -> None:
    transitions = {
        _rewrite_backward_kernel(capture, edit)
        for capture in (certificate["candidate"], certificate["control"])
    }
    if len(transitions) != 1:
        raise AssertionError("candidate/control backward kernel identities differ")
    old_id, new_id = transitions.pop()
    for control in certificate["controls"].values():
        analysis = control.get("analysis", control)
        for write in analysis["borrowed_writes"]:
            if write["kernel_id"] == old_id:
                write["kernel_id"] = new_id


def _rewrite_backward_kernel(capture: dict[str, Any], edit) -> tuple[str, str]:
    old_id = _hop(capture)["kwargs"]["kernel"]
    kernel = next(kernel for kernel in capture["kernels"] if kernel["id"] == old_id)
    edit(kernel)
    kernel["source_sha256"] = sha256_bytes(kernel["source"].encode("utf-8"))
    kernel["id"] = sha256_json(_kernel_identity(kernel))
    for graph in (capture["forward"], capture["backward"]):
        for node in graph["nodes"]:
            if node["target"] == "triton.functional" and node["kwargs"].get("kernel") == old_id:
                node["kwargs"]["kernel"] = kernel["id"]
    return old_id, kernel["id"]


def _digest_token(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


def _target_file_entries(target: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = target.get("files")
    if not isinstance(manifest, list) or not all(isinstance(value, dict) for value in manifest):
        raise AssertionError("target has no closed packet file manifest")
    return manifest


def _target_source_entry(target: dict[str, Any]) -> dict[str, Any]:
    entries = [entry for entry in _target_file_entries(target) if entry.get("role") == "source"]
    if not entries:
        raise AssertionError("target does not have a source manifest entry")
    # The closure mutation only needs one source file; replay checks the full manifest.
    return entries[0]


def _target_dependency_entry(target: dict[str, Any]) -> dict[str, Any]:
    entries = [entry for entry in _target_file_entries(target) if entry.get("role") != "source"]
    if not entries:
        raise AssertionError("target has no non-source packet file manifest entry")
    return entries[-1]


def _update_target_digest(target: dict[str, Any], role: str, digest: str) -> None:
    """Mutate one source or non-source manifest entry for closure testing."""
    if role not in {"source", "dependency"}:
        raise AssertionError(f"unsupported target packet role {role!r}")
    entry = _target_source_entry(target) if role == "source" else _target_dependency_entry(target)
    entry["sha256"] = digest


def _refresh_mutation_closure(
    certificate: dict[str, Any], *, analyze_fn: Callable = analyze
) -> None:
    candidate = Capture.from_dict(certificate["candidate"])
    control_raw = certificate.get("control")
    control = None if control_raw is None else Capture.from_dict(control_raw)
    certificate["candidate"] = candidate.to_dict()
    if control is not None:
        certificate["control"] = control.to_dict()
    certificate["analysis"] = analyze_fn(candidate, control).to_dict()


def _refresh_rejected_source_analysis(certificate: dict[str, Any]) -> None:
    certificate["analysis"] = {
        "outcome": Outcome.UNKNOWN.value,
        "reasons": [Reason.MEMORY_MODEL.value],
        "borrowed_writes": [],
        "collapse": None,
        "unknown_detail": ("MEMORY_MODEL: unsupported memory-model atomic effect tl.atomic_cas"),
        "candidate_digest": sha256_json(certificate["candidate"]),
        "control_digest": (
            sha256_json(certificate["control"]) if certificate.get("control") is not None else None
        ),
    }


def _erase_runtime_witness(certificate: dict[str, Any]) -> None:
    for name in ("candidate", "control"):
        occurrence = _runtime_occurrence(certificate, name)
        if occurrence is not None:
            occurrence["cold_ms"] = 0.0


def _zero_runtime_cold_time(certificate: dict[str, Any]) -> None:
    occurrence = _runtime_occurrence(certificate, "candidate")
    if occurrence is None:
        raise AssertionError("candidate runtime occurrence is required")
    occurrence["cold_ms"] = 0.0


def _runtime_occurrence(certificate: dict[str, Any], name: str) -> dict[str, Any] | None:
    lane = certificate["runtime"][name]
    if lane is None:
        return None
    return lane["compiled"]["occurrence"]


def _make_benchmark_nonpositive(certificate: dict[str, Any]) -> None:
    for lane in certificate["benchmark"]["lanes"].values():
        lane["device_ms"] = [0.0] * len(lane["device_ms"])
        lane["wall_ms"] = [0.0] * len(lane["wall_ms"])
        lane["device_median_ms"] = 0.0
        lane["wall_median_ms"] = 0.0
        lane["device_mad_ms"] = 0.0


def _fabricate_validation_counts(certificate: dict[str, Any]) -> None:
    validation = certificate["validation"]
    validation["oracle_cases"] += 1
    validation["oracle_passed"] = validation["oracle_cases"]
    validation["mutation_cases"] += 1
    validation["mutations_killed"] = validation["mutation_cases"] + 1
    validation["ablations"] += 1
    validation["ablations_observed"] = validation["ablations"]


def _corrupt_hop_binding(capture: dict[str, Any]) -> None:
    hop = _hop(capture)
    bindings = hop["kwargs"]["bindings"]
    for role in _backward_kernel(capture)["reads"]:
        current = bindings.get(role)
        for replacement_role in sorted(bindings):
            replacement = bindings[replacement_role]
            if replacement != current:
                bindings[role] = copy.deepcopy(replacement)
                return
    read_roles = _backward_kernel(capture)["reads"]
    if not read_roles:
        raise AssertionError("backward HOP has no read binding to corrupt")
    bindings[read_roles[0]] = _alternate_tensor_reference(capture, hop)
