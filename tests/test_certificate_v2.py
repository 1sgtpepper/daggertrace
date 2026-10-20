from __future__ import annotations

import copy

import pytest

from daggertrace.certificate import write_certificate
from daggertrace.model import Capture, Outcome
from daggertrace.ownership import analyze
from daggertrace.replay import PROFILE, ReplayError, replay_certificate
from daggertrace.replay_graph import _kernel_runtime_record
from daggertrace.replay_packet import _control_evidence, _runtime_evidence, _sha
from daggertrace.replay_v2 import _capture
from daggertrace.validation_v2_fixtures import _micrograph
from daggertrace.version import CHECKER_PROFILE

from _certificate_fixtures import (
    GENERIC_REPRODUCER,
    _packet,
    _with_second_differentiable_input,
    _write_canonical,
)
def _three_hop_candidate(capture: Capture) -> Capture:
    raw = capture.to_dict()
    graph = raw["backward"]
    original = graph["nodes"]
    tangent = copy.deepcopy(original[0])
    view = copy.deepcopy(original[1])
    allocation = copy.deepcopy(original[2])
    hop = copy.deepcopy(original[3])
    getitem = copy.deepcopy(original[4])
    add = copy.deepcopy(original[5])
    output = copy.deepcopy(original[6])
    nodes = [tangent, view]
    current = "n1"
    read_role, write_role = hop["kwargs"]["bindings"]
    for index in range(3):
        allocation_node = copy.deepcopy(allocation)
        allocation_id = f"n{len(nodes)}"
        allocation_node["id"] = allocation_id
        allocation_node["args"] = {"tuple": [{"ref": current}]}
        allocation_node["tensor"]["alias_group"] = f"destination-{index}"
        nodes.append(allocation_node)

        hop_node = copy.deepcopy(hop)
        hop_id = f"n{len(nodes)}"
        hop_node["id"] = hop_id
        hop_node["kwargs"]["bindings"] = {
            read_role: {"ref": current},
            write_role: {"ref": allocation_id},
        }
        nodes.append(hop_node)

        getitem_node = copy.deepcopy(getitem)
        getitem_id = f"n{len(nodes)}"
        getitem_node["id"] = getitem_id
        getitem_node["args"] = {"tuple": [{"ref": hop_id}, write_role]}
        getitem_node["tensor"] = copy.deepcopy(allocation_node["tensor"])
        getitem_node["tensor"]["alias_group"] = f"functional-{index}"
        nodes.append(getitem_node)
        current = getitem_id

    add["id"] = f"n{len(nodes)}"
    add["args"] = {"tuple": [{"ref": "n0"}, {"ref": current}]}
    add["tensor"]["alias_group"] = "gradient"
    nodes.append(add)
    output["id"] = f"n{len(nodes)}"
    output["args"] = {"tuple": [{"tuple": [{"ref": add["id"]}]}]}
    nodes.append(output)
    graph["nodes"] = nodes
    return Capture.from_dict(raw)


def test_nullable_control_sidecar_and_candidate_only_safe_replay(tmp_path):
    certificate_path, certificate = _packet(tmp_path)

    assert certificate["control"] is None
    assert (tmp_path / "control.capture.json").read_bytes() == b"null\n"
    result = replay_certificate(certificate_path)

    assert result["valid"] is True
    assert result["outcome"] == "SAFE_IN_FRAGMENT"
    assert result["contextual_collapse"] is False


@pytest.mark.parametrize(
    ("edit", "analyzer_detail", "replay_detail"),
    [
        (
            lambda raw: raw["diagnostics"]["inputs"].append(
                copy.deepcopy(raw["diagnostics"]["inputs"][0])
            ),
            "capture diagnostics inputs are malformed",
            "do not match runtime inputs",
        ),
        (
            lambda raw: raw["diagnostics"]["inputs"].pop(),
            "capture diagnostics inputs are malformed",
            "do not match runtime inputs",
        ),
        (
            lambda raw: raw["diagnostics"]["inputs"][0].update(requires_grad=1),
            "capture diagnostics requires_grad must be boolean",
            r"candidate\.diagnostics\.inputs\[0\] is malformed",
        ),
        (
            lambda raw: raw["diagnostics"].pop("grad_output"),
            "capture diagnostics fields are malformed",
            r"candidate\.diagnostics fields changed",
        ),
        (
            lambda raw: raw["diagnostics"].update(extra=None),
            "capture diagnostics fields are malformed",
            r"candidate\.diagnostics fields changed",
        ),
        (
            lambda raw: raw["diagnostics"]["inputs"][0].pop("shape"),
            "capture diagnostics inputs[0] is malformed",
            r"candidate\.diagnostics\.inputs\[0\] fields changed",
        ),
        (
            lambda raw: raw["diagnostics"]["inputs"][0].update(shape=[True, 2]),
            "capture diagnostics inputs[0] layout is malformed",
            r"candidate\.diagnostics\.inputs\[0\]\.shape\[0\] must be an integer",
        ),
        (
            lambda raw: raw["diagnostics"]["grad_output"].pop("dtype"),
            "capture diagnostics grad_output is malformed",
            r"candidate\.diagnostics\.grad_output fields changed",
        ),
        (
            lambda raw: raw["diagnostics"]["grad_output"].update(storage_offset=-1),
            "capture diagnostics grad_output storage offset is malformed",
            r"candidate\.diagnostics\.grad_output\.storage_offset is below its minimum 0",
        ),
        (
            lambda raw: (
                raw["backward"]["nodes"][-1]["args"]["tuple"][0]["tuple"].reverse(),
                raw["backward"]["output_roles"].reverse(),
            ),
            "backward gradient projection does not match differentiable inputs",
            "gradient outputs",
        ),
    ],
)
def test_v2_diagnostics_input_admission_matches_public_replay(
    tmp_path, edit, analyzer_detail, replay_detail
):
    certificate_path, certificate = _packet(tmp_path)
    candidate = _with_second_differentiable_input(
        Capture.from_dict(certificate["candidate"])
    ).to_dict()
    certificate["candidate"] = candidate
    certificate["analysis"] = analyze(Capture.from_dict(candidate)).to_dict()
    certificate["digest"] = _sha({key: value for key, value in certificate.items() if key != "digest"})
    _write_canonical(tmp_path / "candidate.capture.json", candidate)
    _write_canonical(certificate_path, certificate)
    assert replay_certificate(certificate_path)["valid"] is True
    edit(candidate)

    result = analyze(Capture.from_dict(candidate))

    assert result.outcome is Outcome.UNKNOWN
    assert result.unknown_detail == analyzer_detail
    certificate["digest"] = _sha({key: value for key, value in certificate.items() if key != "digest"})
    _write_canonical(tmp_path / "candidate.capture.json", candidate)
    _write_canonical(certificate_path, certificate)
    with pytest.raises(ReplayError, match=replay_detail):
        replay_certificate(certificate_path)


def test_certificate_writer_rejects_legacy_adapter_metadata(tmp_path):
    packet = tmp_path / "legacy"
    certificate_path, certificate = _packet(packet)
    certificate["target"] = {
        "name": "adapter provenance",
        "repository": "https://example.invalid/repository",
        "revision": "b" * 40,
        "specialization": {"case": "safe"},
    }
    with pytest.raises(ValueError, match="closed generic manifest"):
        write_certificate(certificate_path, certificate, reproducer_bytes=GENERIC_REPRODUCER)


def test_certificate_writer_rejects_retired_certificate_schema(tmp_path):
    certificate_path, certificate = _packet(tmp_path)
    certificate["schema"] = "daggertrace.aot-storage-ownership/3"
    certificate["digest"] = _sha(
        {key: value for key, value in certificate.items() if key != "digest"}
    )

    with pytest.raises(ValueError, match="unsupported certificate schema"):
        write_certificate(certificate_path, certificate, reproducer_bytes=GENERIC_REPRODUCER)


def test_candidate_only_violation_is_a_valid_certificate_result(tmp_path):
    certificate_path, certificate = _packet(tmp_path, alias_destination=True)

    assert certificate["control"] is None
    result = replay_certificate(certificate_path)

    assert result["outcome"] == "VIOLATION"
    assert result["contextual_collapse"] is False


def test_three_hop_packet_shape_replays_without_a_fixed_hop_count(tmp_path):
    certificate_path, certificate = _packet(tmp_path)
    candidate = _three_hop_candidate(
        Capture.from_dict(copy.deepcopy(certificate["candidate"]))
    )
    certificate["candidate"] = candidate.to_dict()
    certificate["analysis"] = analyze(candidate, None).to_dict()
    certificate["digest"] = _sha({key: value for key, value in certificate.items() if key != "digest"})
    _write_canonical(tmp_path / "candidate.capture.json", certificate["candidate"])
    _write_canonical(certificate_path, certificate)

    result = replay_certificate(certificate_path)

    assert result["valid"] is True
    assert result["outcome"] == "SAFE_IN_FRAGMENT"


def test_paired_control_and_noncanonical_null_sidecar_are_distinct_states(tmp_path):
    certificate_path, certificate = _packet(tmp_path, paired=True)
    assert replay_certificate(certificate_path)["valid"] is True

    certificate["control"] = None
    certificate["digest"] = _sha({key: value for key, value in certificate.items() if key != "digest"})
    _write_canonical(certificate_path, certificate)

    with pytest.raises(ReplayError, match="control.capture.json differs"):
        replay_certificate(certificate_path)


def test_runtime_wrapper_identity_matches_the_closed_core_profile():
    _kernel_runtime_record(
        {
            "wrapper_type": "triton.runtime.autotuner.Autotuner",
            "leaf_type": "triton.runtime.jit.JITFunction",
            "config": {
                "kwargs": {},
                "num_warps": 8,
                "num_ctas": 1,
                "num_stages": 3,
                "maxnreg": None,
                "pre_hook": None,
                "ir_override": None,
            },
            "jit_cache_key": "a" * 64,
        }
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_warps", 64, "exceeds its maximum"),
        ("num_warps", 3, "positive power of two"),
    ],
)
def test_runtime_config_matches_the_closed_core_grammar(field, value, message):
    runtime = {
        "wrapper_type": "triton.runtime.autotuner.Autotuner",
        "leaf_type": "triton.runtime.jit.JITFunction",
        "config": {
            "kwargs": {},
            "num_warps": 8,
            "num_ctas": 1,
            "num_stages": 3,
            "maxnreg": None,
            "pre_hook": None,
            "ir_override": None,
        },
        "jit_cache_key": "a" * 64,
    }
    runtime["config"][field] = value

    with pytest.raises(ReplayError, match=message):
        _kernel_runtime_record(runtime)


def test_runtime_config_rejects_noncanonical_wrapper_and_nonnull_maxnreg():
    runtime = {
        "wrapper_type": "observed.wrapper",
        "leaf_type": "triton.runtime.jit.JITFunction",
        "config": {
            "kwargs": {},
            "num_warps": 8,
            "num_ctas": 1,
            "num_stages": 3,
            "maxnreg": None,
            "pre_hook": None,
            "ir_override": None,
        },
        "jit_cache_key": "a" * 64,
    }
    with pytest.raises(ReplayError, match="wrapper identity"):
        _kernel_runtime_record(runtime)

    runtime["wrapper_type"] = "triton.runtime.autotuner.Autotuner"
    runtime["config"]["maxnreg"] = 1
    with pytest.raises(ReplayError, match="hooks and IR overrides"):
        _kernel_runtime_record(runtime)


def _runtime_environment():
    return {
        "python": "3.14.6",
        "platform": "test",
        "torch": "2.12.1+cu130",
        "torch_git": "7269437d655783a26cba32aa88195b741ff496aa",
        "triton": "3.7.1",
        "cuda_runtime": "13.0",
        "device": "observed-device",
        "capability": [1, 0],
        "total_memory_bytes": 0,
        "multiprocessors": 0,
        "driver": None,
    }


def _runtime_occurrence():
    return {
        "cold_ms": 1.0,
        "generated_kernel_count": 3,
        "ir_nodes_pre_fusion": 7,
        "counter_delta": {
            "aot_autograd": {"total": 1, "ok": 1},
            "inductor": {"extern_calls": 1},
        },
    }


def test_runtime_evidence_accepts_candidate_only_with_nullable_control_lane():
    _runtime_evidence(
        {
            "environment": _runtime_environment(),
            "candidate": {"compiled": {"occurrence": _runtime_occurrence()}},
            "control": None,
        }
    )
    _runtime_evidence(
        {
            "environment": _runtime_environment(),
            "candidate": {"compiled": None},
            "control": None,
        }
    )


def test_runtime_evidence_accepts_a_paired_control_occurrence():
    _runtime_evidence(
        {
            "environment": _runtime_environment(),
            "candidate": {"compiled": {"occurrence": _runtime_occurrence()}},
            "control": {"compiled": {"occurrence": _runtime_occurrence()}},
        }
    )


def test_runtime_evidence_rejects_target_numerical_observations_in_common_fields():
    runtime = {
        "environment": _runtime_environment(),
        "candidate": {"compiled": None},
        "control": None,
        "candidate_vs_reference": {"target_metric": 0.0},
    }

    with pytest.raises(ReplayError, match="runtime fields changed"):
        _runtime_evidence(runtime)


def test_runtime_evidence_rejects_liger_shaped_compiled_comparisons():
    runtime = {
        "environment": _runtime_environment(),
        "candidate": {
            "compiled": {
                "occurrence": _runtime_occurrence(),
                "candidate_vs_reference": {},
            }
        },
        "control": None,
    }

    with pytest.raises(ReplayError, match="runtime.candidate.compiled fields changed"):
        _runtime_evidence(runtime)


def test_runtime_evidence_rejects_missing_candidate_lane():
    with pytest.raises(ReplayError, match="runtime.candidate is required"):
        _runtime_evidence({"environment": _runtime_environment(), "candidate": None, "control": None})


@pytest.mark.parametrize(
    ("field", "value"),
    (("torch", "2.12.1+test"), ("torch_git", "c" * 40)),
)
def test_runtime_evidence_requires_exact_pinned_torch_identity(field, value):
    environment = _runtime_environment()
    environment[field] = value

    with pytest.raises(ReplayError, match="outside the versioned framework profile"):
        _runtime_evidence(
            {"environment": environment, "candidate": {"compiled": None}, "control": None}
        )


def test_control_evidence_accepts_only_generic_analysis_records():
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    analysis = analyze(candidate, None).to_dict()

    _control_evidence({"candidate-only-check": analysis})
    _control_evidence({"candidate-only-check": {"analysis": analysis}})


def test_control_evidence_rejects_target_numerical_comparisons():
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    analysis = analyze(candidate, None).to_dict()

    with pytest.raises(ReplayError, match="controls.target-check fields changed"):
        _control_evidence(
            {
                "target-check": {
                    "analysis": analysis,
                    "candidate_vs_control": None,
                }
            }
        )


def test_replay_and_core_share_the_whole_storage_profile():
    assert PROFILE == CHECKER_PROFILE == "torch-2.12-triton-hop-whole-storage/2"


@pytest.mark.parametrize(
    ("field", "value"),
    [("torch", "2.12.1+test"), ("torch_git", "c" * 40)],
)
def test_framework_profile_requires_exact_pinned_torch_identity(field, value):
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    raw = candidate.to_dict()
    raw["framework"][field] = value

    with pytest.raises(ReplayError, match="outside the versioned profile"):
        _capture(raw, "candidate")
