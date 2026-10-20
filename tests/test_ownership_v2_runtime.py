from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from daggertrace import ownership_v2_runtime
from daggertrace.canonical import sha256_bytes, sha256_json
from daggertrace.model import Capture, Outcome, Reason
from daggertrace.ownership import analyze
from daggertrace.validation_v2_fixtures import _micrograph
from _ownership_fixtures import (
    _constant_signature,
    _edited,
    _forward_with_empty,
    _insert_connected_clone_before_accumulation,
    _kernel_identity,
    _kernel_for_capture,
    _node,
    _output_node,
    _paired_allocation_outputs,
    _read_only_hop_raw,
    _redirect_accumulation_to_existing_empty,
    _redirect_hop_write_to_tangent,
    _rename_kernel_formals,
    _retarget_allocation,
    _tangent_node,
    _tangent_nodes,
    _two_input_raw,
    _two_tangent_graphs,
    _append_accumulation,
    _with_explicit_saved_output,
)

def test_direct_residual_proves_borrowed_write_and_collapse():
    candidate, control = _micrograph(True, "direct", False, 2, True)
    result = analyze(candidate, control)
    assert result.outcome == Outcome.VIOLATION
    assert result.borrowed_writes[0].before_version == 0
    assert result.borrowed_writes[0].after_version == 1
    assert result.collapse is not None
    assert result.collapse.expected_parts[0] == "tangent:output:0"
    assert len(set(result.collapse.actual_parts)) == 1

def test_v2_model_rejects_a_read_only_hop():
    raw = _read_only_hop_raw()
    result = analyze(Capture.from_dict(raw))

    assert result.outcome is Outcome.UNKNOWN
    assert "writes differ" in result.unknown_detail

def test_v2_gradient_projection_matches_two_differentiable_inputs():
    valid = _two_input_raw()
    assert analyze(Capture.from_dict(valid)).outcome is Outcome.SAFE_IN_FRAGMENT

    first_only = copy.deepcopy(valid)
    first_only["diagnostics"]["inputs"][1]["requires_grad"] = False
    first_only["backward"]["nodes"][-1]["args"]["tuple"][0]["tuple"].pop()
    first_only["backward"]["output_roles"].pop()
    assert analyze(Capture.from_dict(first_only)).outcome is Outcome.SAFE_IN_FRAGMENT

    missing = copy.deepcopy(valid)
    output = missing["backward"]["nodes"][-1]
    output["args"]["tuple"][0]["tuple"].pop()
    missing["backward"]["output_roles"].pop()
    assert analyze(Capture.from_dict(missing)).outcome is Outcome.UNKNOWN

    reordered = copy.deepcopy(valid)
    reordered["backward"]["nodes"][-1]["args"]["tuple"][0]["tuple"].reverse()
    reordered["backward"]["output_roles"].reverse()
    result = analyze(Capture.from_dict(reordered))
    assert result.outcome is Outcome.UNKNOWN
    assert result.unknown_detail == "backward gradient projection does not match differentiable inputs"

def test_paired_control_with_borrowed_write_is_unknown_but_canonical_pair_is_valid():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    canonical = analyze(candidate, control)
    assert canonical.outcome == Outcome.VIOLATION
    assert canonical.reasons == (
        Reason.BORROWED_COTANGENT_WRITE,
        Reason.CONTRIBUTION_COLLAPSE,
    )
    assert canonical.collapse is not None

    malicious_control = _redirect_hop_write_to_tangent(control)
    assert ownership_v2_runtime._run_backward(malicious_control).borrowed_writes

    result = analyze(candidate, malicious_control)
    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.TWIN_RELATION,)
    assert result.collapse is None

def test_paired_twin_rejects_a_connected_clone_outside_the_allowed_substitution():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    raw = control.to_dict()
    _insert_connected_clone_before_accumulation(raw)

    result = analyze(candidate, Capture.from_dict(raw))

    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.TWIN_RELATION,)
    assert "normalized backward operator skeletons differ" in result.unknown_detail

def test_candidate_only_borrowed_write_remains_a_violation_without_collapse():
    candidate, _ = _micrograph(True, "direct", False, 1, False)

    result = analyze(candidate)

    assert result.outcome == Outcome.VIOLATION
    assert result.reasons == (Reason.BORROWED_COTANGENT_WRITE,)
    assert result.borrowed_writes
    assert result.collapse is None
    assert result.control_digest is None

def test_out_of_place_destination_is_safe_in_fragment():
    candidate, control = _micrograph(False, "direct", False, 1, False)
    result = analyze(candidate, control)
    assert result.outcome == Outcome.SAFE_IN_FRAGMENT
    assert result.borrowed_writes == ()
    assert result.collapse is None

@pytest.mark.parametrize("target", ["aten.empty_like", "aten.zeros"])
def test_new_fresh_allocation_forms_are_candidate_only_safe(target):
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    normalized = _retarget_allocation(candidate, target)

    result = analyze(normalized)

    assert result.outcome == Outcome.SAFE_IN_FRAGMENT
    assert result.reasons == (Reason.FRESH_DESTINATION,)
    assert result.control_digest is None

def test_new_fresh_allocation_forms_reject_the_opposite_argument_shape():
    candidate, _ = _micrograph(False, "direct", False, 1, False)

    for target, argument in (
        ("aten.empty_like", [1]),
        ("aten.zeros", {"ref": "n0"}),
    ):
        result = analyze(
            _edited(
                candidate,
                lambda raw, target=target, argument=argument: _node(
                    raw, "aten.zeros_like"
                ).update(target=target, args={"tuple": [argument]}),
            )
        )

        assert result.outcome == Outcome.UNKNOWN
        assert result.reasons == (Reason.GRAPH_EFFECT,)

def test_normalized_pure_calls_and_eight_warp_kernel_are_candidate_only_safe():
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    raw = copy.deepcopy(candidate.to_dict())
    allocation = _node(raw, "aten.zeros_like")
    allocation.update(target="aten.empty_like", args={"tuple": [{"ref": "n0"}]})
    kernel = _kernel_for_capture(raw)
    kernel["source"] = """@triton.jit
def renamed_kernel(dY_ptr, dX_ptr):
    value = tl.load(dY_ptr)
    value = tl.exp(value)
    peak = tl.max(value, axis=0)
    for _ in tl.static_range(0, 1):
        value = tl.where(value > peak, value, float("-inf"))
    tl.store(dX_ptr, value)
"""
    kernel["source_sha256"] = sha256_bytes(kernel["source"].encode("utf-8"))
    kernel["argument_names"] = ["dY_ptr", "dX_ptr"]
    kernel["reads"] = ["dY_ptr"]
    kernel["writes"] = ["dX_ptr"]
    kernel["source_reads"] = ["dY_ptr"]
    kernel["source_writes"] = ["dX_ptr"]
    kernel["runtime"]["config"]["num_warps"] = 8
    kernel["id"] = sha256_json(_kernel_identity(kernel))
    for node in raw["backward"]["nodes"]:
        if node["target"] == "triton.functional":
            node["kwargs"]["kernel"] = kernel["id"]
            node["kwargs"]["bindings"] = {
                "dY_ptr": node["kwargs"]["bindings"]["read_role_0"],
                "dX_ptr": node["kwargs"]["bindings"]["write_role_0"],
            }
            node["kwargs"]["writes"] = ["dX_ptr"]
        elif node["target"] == "operator.getitem":
            node["args"]["tuple"][1] = "dX_ptr"

    result = analyze(Capture.from_dict(raw))

    assert result.outcome == Outcome.SAFE_IN_FRAGMENT
    assert result.reasons == (Reason.FRESH_DESTINATION,)

def test_atomic_protocol_is_unknown_with_a_memory_model_reason():
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    source = """def protocol(dY_ptr, dX_ptr):
    while True:
        old = tl.atomic_cas(dY_ptr, 0, 1)
    tl.debug_barrier()
    tl.atomic_xchg(dY_ptr, 0)
    tl.store(dX_ptr, old)
"""
    kernel = replace(
        candidate.kernels[0],
        source=source,
        source_sha256=sha256_bytes(source.encode("utf-8")),
        argument_names=("dY_ptr", "dX_ptr"),
        reads=("dY_ptr",),
        writes=("dX_ptr",),
        source_reads=("dY_ptr",),
        source_writes=("dX_ptr",),
    )

    result = analyze(replace(candidate, kernels=(kernel,)))

    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.MEMORY_MODEL,)

def test_distinct_hop_occurrences_preserve_multiplicity_with_graph_ordinals():
    candidate, _ = _micrograph(True, "none", False, 1, False)
    raw = copy.deepcopy(candidate.to_dict())
    nodes = raw["backward"]["nodes"]
    output = nodes.pop()
    first_hop = next(node for node in nodes if node["target"] == "triton.functional")
    first_result = next(node for node in nodes if node["target"] == "operator.getitem")

    second_hop = copy.deepcopy(first_hop)
    second_hop["id"] = "n4"
    second_result = copy.deepcopy(first_result)
    second_result["id"] = "n5"
    second_result["args"]["tuple"][0] = {"ref": "n4"}
    second_result["tensor"]["alias_group"] = "functional-1"

    accumulation = copy.deepcopy(first_result)
    accumulation.update(
        id="n6",
        op="call_function",
        target="aten.add",
        args={"tuple": [{"ref": first_result["id"]}, {"ref": second_result["id"]}]},
        kwargs={},
        role=None,
        tensor=copy.deepcopy(first_result["tensor"]),
        gradient_accumulation=True,
    )
    accumulation["tensor"]["alias_group"] = "gradient-twinned"

    output["id"] = "n7"
    output["args"] = {"tuple": [{"tuple": [{"ref": "n6"}]}]}
    nodes.extend([second_hop, second_result, accumulation, output])
    duplicated = Capture.from_dict(raw)

    result = analyze(duplicated)
    graph_result = ownership_v2_runtime._run_backward(duplicated)

    assert result.outcome == Outcome.VIOLATION
    assert len(graph_result.borrowed_writes) == 2
    # Both HOPs bind the original SSA input, not the preceding functional result.
    assert [
        (write.before_version, write.after_version)
        for write in graph_result.borrowed_writes
    ] == [(0, 1), (0, 1)]
    parts = graph_result.gradients[0].parts
    assert len(parts) == 2
    assert parts[0] != parts[1]

def test_hop_accepts_a_coherent_synthetic_constant_signature():
    candidate, _ = _micrograph(False, "direct", False, 1, False)

    assert analyze(_constant_signature(candidate)).outcome == Outcome.SAFE_IN_FRAGMENT

@pytest.mark.parametrize(
    "constant_case", ["missing", "extra", "overlap", "tensor_ref"],
    ids=["missing", "extra", "overlap", "tensor_ref"],
)
def test_hop_rejects_invalid_constant_signatures(constant_case):
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    synthetic = _constant_signature(candidate)
    kernel = _kernel_for_capture(synthetic)
    if constant_case == "missing":
        constants = {}
    elif constant_case == "extra":
        constants = {"BLOCK_SIZE": 128, "EXTRA": 1}
    elif constant_case == "overlap":
        constants = {"BLOCK_SIZE": 128, kernel["reads"][0]: 1}
    else:
        constants = {"BLOCK_SIZE": {"ref": "n0"}}
    result = analyze(
        _edited(
            synthetic,
            lambda raw: _node(raw, "triton.functional")["kwargs"].update(constants=constants),
        )
    )

    assert result.outcome == Outcome.UNKNOWN

def test_sole_tangent_role_erased_is_unknown():
    candidate, _ = _micrograph(True, "direct", False, 1, False)

    result = analyze(
        _edited(candidate, lambda raw: _tangent_node(raw).update(role=None))
    )

    assert result.outcome == Outcome.UNKNOWN

def test_role_null_saved_placeholder_remains_admitted_with_a_tangent():
    candidate, _ = _micrograph(False, "direct", False, 1, True)

    result = analyze(
        _edited(
            candidate,
            lambda raw: _node(raw, "aten.empty").update(
                op="placeholder",
                target="placeholder",
                args={"tuple": []},
                kwargs={},
                role=None,
            ),
        )
    )

    assert result.outcome == Outcome.SAFE_IN_FRAGMENT


def test_explicit_saved_output_placeholder_matches_the_forward_saved_domain():
    candidate, _ = _micrograph(False, "none", False, 1, False)
    raw = _with_explicit_saved_output(candidate.to_dict())
    capture = Capture.from_dict(raw)

    result = analyze(capture)
    graph_result = ownership_v2_runtime._run_backward(capture)

    assert result.outcome == Outcome.SAFE_IN_FRAGMENT
    assert graph_result.borrowed_writes == ()


def test_explicit_saved_output_placeholder_rejects_an_undeclared_slot():
    candidate, _ = _micrograph(False, "none", False, 1, False)
    raw = _with_explicit_saved_output(candidate.to_dict())
    saved = next(
        node
        for node in raw["backward"]["nodes"]
        if node["role"] and node["role"].get("kind") == "saved_output"
    )
    saved["role"]["index"] = 0

    result = analyze(Capture.from_dict(raw))

    assert result.outcome == Outcome.UNKNOWN
    assert "outside the saved-output domain" in result.unknown_detail


def test_explicit_saved_output_placeholder_rejects_a_missing_saved_declaration():
    candidate, _ = _micrograph(False, "none", False, 1, False)
    raw = _with_explicit_saved_output(candidate.to_dict())
    raw["forward"]["output_roles"][1] = {"kind": "output", "index": 1}

    result = analyze(Capture.from_dict(raw))

    assert result.outcome == Outcome.UNKNOWN
    assert "outside the saved-output domain" in result.unknown_detail


def test_explicit_saved_output_role_families_remain_closed():
    candidate, _ = _micrograph(False, "none", False, 1, False)

    forward_saved = _with_explicit_saved_output(candidate.to_dict())
    forward_saved["forward"]["nodes"][0]["role"] = {
        "kind": "saved_output",
        "index": 1,
    }
    assert analyze(Capture.from_dict(forward_saved)).outcome == Outcome.UNKNOWN

    tangent_saved = _with_explicit_saved_output(candidate.to_dict())
    tangent_saved["backward"]["nodes"][0]["role"] = {
        "kind": "tangent",
        "output": {"kind": "saved_output", "index": 1},
    }
    assert analyze(Capture.from_dict(tangent_saved)).outcome == Outcome.UNKNOWN


def test_isolated_write_violates_contract_without_contextual_harm():
    candidate, control = _micrograph(True, "none", False, 1, False)
    result = analyze(candidate, control)
    assert result.outcome == Outcome.VIOLATION
    assert result.borrowed_writes
    assert result.collapse is None

def test_materialized_residual_preserves_value_before_write():
    candidate, control = _micrograph(True, "materialized", True, 2, False)
    result = analyze(candidate, control)
    assert result.outcome == Outcome.VIOLATION
    assert result.collapse is None

def test_allocation_contribution_is_scoped_to_its_capture():
    candidate, control = _micrograph(True, "direct", False, 1, True)
    candidate = _redirect_accumulation_to_existing_empty(candidate)
    control = _redirect_accumulation_to_existing_empty(control)

    result = analyze(candidate)
    assert result.outcome == Outcome.VIOLATION
    assert result.reasons == (Reason.BORROWED_COTANGENT_WRITE,)
    assert result.collapse is None

    paired = analyze(candidate, control)
    assert paired.outcome == Outcome.UNKNOWN
    assert paired.reasons == (Reason.TWIN_RELATION,)
    assert paired.collapse is None

def test_fresh_materialized_result_cannot_claim_tangent_storage():
    candidate, _ = _micrograph(False, "materialized", False, 1, False)
    assert analyze(candidate).outcome == Outcome.SAFE_IN_FRAGMENT

    for alias_group in ("tangent", "destination", None):
        result = analyze(
            _edited(
                candidate,
                lambda raw, alias_group=alias_group: _node(raw, "aten.mul")["tensor"].update(
                    alias_group=alias_group
                ),
            )
        )
        assert result.outcome == Outcome.UNKNOWN
        assert result.reasons == (Reason.REGION_ALIAS,)
        assert "fresh result" in result.unknown_detail

def test_forward_fresh_root_collision_fails_closed():
    candidate, _ = _micrograph(False, "direct", False, 1, False)

    result = analyze(_forward_with_empty(candidate, "primal"))

    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.REGION_ALIAS,)
    assert "fresh result" in result.unknown_detail

def test_forward_fresh_root_with_distinct_storage_is_admitted():
    candidate, _ = _micrograph(False, "direct", False, 1, False)

    result = analyze(_forward_with_empty(candidate, "forward-empty"))

    assert result.outcome == Outcome.SAFE_IN_FRAGMENT
    assert result.reasons == (Reason.FRESH_DESTINATION,)

def test_placeholder_storage_identity_is_required():
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    assert analyze(candidate).outcome == Outcome.SAFE_IN_FRAGMENT

    result = analyze(
        _edited(candidate, lambda raw: _tangent_node(raw)["tensor"].update(alias_group=None))
    )
    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.GRAPH_EFFECT,)
    assert "view node" in result.unknown_detail

def test_candidate_only_does_not_invent_erased_prewrite_liveness():
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    result = analyze(candidate)
    assert result.outcome == Outcome.VIOLATION
    assert result.collapse is None

def test_unknown_fx_target_fails_closed():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    raw = copy.deepcopy(candidate.to_dict())
    raw["backward"]["nodes"][1]["target"] = "unsupported:aten.reshape"
    result = analyze(Capture.from_dict(raw), control)
    assert result.outcome == Outcome.UNKNOWN
    assert "unsupported captured targets" in result.unknown_detail

def test_alpha_renamed_kernel_is_not_target_name_dispatched():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    kernel = _kernel_for_capture(candidate)
    assert kernel["qualname"]
    assert analyze(candidate, control).collapse is not None

def test_generic_ownership_is_invariant_under_coherent_kernel_role_alpha_renaming():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    baseline = analyze(candidate, control)

    read_role = "formal_alpha"
    write_role = "formal_omega"
    renamed_candidate = _rename_kernel_formals(candidate, read_role, write_role)
    renamed_control = _rename_kernel_formals(control, read_role, write_role)
    renamed = analyze(renamed_candidate, renamed_control)

    assert baseline.outcome == renamed.outcome == Outcome.VIOLATION
    assert baseline.reasons == renamed.reasons == (
        Reason.BORROWED_COTANGENT_WRITE,
        Reason.CONTRIBUTION_COLLAPSE,
    )
    assert len(baseline.borrowed_writes) == len(renamed.borrowed_writes) == 1

    baseline_write = baseline.borrowed_writes[0]
    renamed_write = renamed.borrowed_writes[0]
    assert renamed_write.tangent == baseline_write.tangent == "tangent:output:0"
    assert renamed_write.hop_node == baseline_write.hop_node
    assert renamed_write.read_role == read_role
    assert renamed_write.write_role == write_role
    renamed_kernel = _kernel_for_capture(renamed_candidate)
    baseline_kernel = _kernel_for_capture(candidate)
    assert renamed_write.kernel_id == renamed_kernel["id"]
    assert (renamed_write.before_version, renamed_write.after_version) == (0, 1)

    assert baseline.collapse is not None
    assert renamed.collapse is not None
    assert (
        renamed.collapse.gradient_of,
        renamed.collapse.accumulation_node,
        renamed.collapse.replaced_part,
    ) == (
        baseline.collapse.gradient_of,
        baseline.collapse.accumulation_node,
        baseline.collapse.replaced_part,
    )
    assert renamed.collapse.expected_parts == (
        "tangent:output:0",
        renamed.collapse.replacement_part,
    )
    assert renamed.collapse.actual_parts == (
        renamed.collapse.replacement_part,
        renamed.collapse.replacement_part,
    )

    for capture in (renamed_candidate, renamed_control):
        kernel = _kernel_for_capture(capture)
        assert tuple(kernel["argument_names"][:2]) == (read_role, write_role)
        assert kernel["reads"] == kernel["source_reads"] == [read_role]
        assert kernel["writes"] == kernel["source_writes"] == [write_role]
        assert all(role not in kernel["source"] for role in baseline_kernel["argument_names"])
        assert kernel["source_sha256"] != baseline_kernel["source_sha256"]
        hop = next(node for node in capture.backward.nodes if node.target == "triton.functional")
        assert hop.kwargs["kernel"] == kernel["id"]
        assert set(hop.kwargs["bindings"]) == {read_role, write_role}
        assert hop.kwargs["writes"] == [write_role]
        getitem = next(node for node in capture.backward.nodes if node.target == "operator.getitem")
        assert getitem.args["tuple"][1] == write_role

    assert _kernel_for_capture(renamed_candidate)["id"] == _kernel_for_capture(renamed_control)["id"]
    assert _kernel_for_capture(renamed_candidate)["id"] != baseline_kernel["id"]

def test_paired_collapse_is_invariant_to_backward_capture_local_alias_group_names():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    baseline = analyze(candidate, control)
    raw = control.to_dict()
    for node in raw["backward"]["nodes"]:
        tensor = node.get("tensor")
        if tensor is not None and tensor["alias_group"] is not None:
            tensor["alias_group"] = f"renamed-{tensor['alias_group']}"

    renamed = analyze(candidate, Capture.from_dict(raw))

    assert renamed.outcome == baseline.outcome == Outcome.VIOLATION
    assert renamed.reasons == baseline.reasons
    assert renamed.borrowed_writes == baseline.borrowed_writes
    assert renamed.collapse == baseline.collapse


def test_paired_collapse_requires_exact_forward_alias_labels():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    raw = control.to_dict()
    for node in raw["forward"]["nodes"]:
        if node["tensor"] is not None:
            node["tensor"]["alias_group"] = f"renamed-{node['tensor']['alias_group']}"

    result = analyze(candidate, Capture.from_dict(raw))

    assert result.outcome is Outcome.UNKNOWN
    assert result.reasons == (Reason.TWIN_RELATION,)
    assert "forward graphs differ" in result.unknown_detail


def test_resealed_paired_allocations_do_not_share_a_coarse_contribution_part():
    candidate, control = _paired_allocation_outputs()

    result = analyze(candidate, control)

    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.TWIN_RELATION,)

def test_fresh_allocation_and_clone_require_new_captured_storage_identity():
    candidate, _ = _micrograph(False, "direct", False, 1, False)

    for target in ("aten.zeros_like", "aten.clone"):
        def retarget(raw, *, target=target, alias_group="destination"):
            node = _node(raw, "aten.zeros_like")
            node["target"] = target
            node["tensor"]["alias_group"] = alias_group

        assert analyze(_edited(candidate, retarget)).outcome == Outcome.SAFE_IN_FRAGMENT

        for alias_group in (None, "tangent"):
            result = analyze(
                _edited(
                    candidate,
                    lambda raw, target=target, alias_group=alias_group: retarget(
                        raw, target=target, alias_group=alias_group
                    ),
                )
            )
            assert result.outcome == Outcome.UNKNOWN
            assert result.reasons == (Reason.REGION_ALIAS,)
            assert "fresh result" in result.unknown_detail

def test_paired_control_rejects_resealed_allocation_alias_collision():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    malicious = _edited(
        control,
        lambda raw: _node(raw, "aten.zeros_like")["tensor"].update(alias_group="tangent"),
    )

    result = analyze(candidate, malicious)
    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.REGION_ALIAS,)
    assert result.collapse is None

def test_gradient_output_slots_reject_bool_negative_out_of_domain_and_duplicates():
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    for index in (True, -1, 1):
        result = analyze(
            _edited(
                candidate,
                lambda raw, index=index: raw["backward"]["output_roles"][0]["primal"].update(
                    index=index
                ),
            )
        )
        assert result.outcome == Outcome.UNKNOWN

    def duplicate_slot(raw):
        backward = raw["backward"]
        outputs = _output_node(raw)["args"]["tuple"][0]["tuple"]
        outputs.append(copy.deepcopy(outputs[0]))
        backward["output_roles"].append(copy.deepcopy(backward["output_roles"][0]))

    duplicate = analyze(_edited(candidate, duplicate_slot))
    assert duplicate.outcome == Outcome.UNKNOWN
    assert "duplicate gradient output slot" in duplicate.unknown_detail

def test_collapse_preserves_the_exact_tangent_role_with_multiple_tangents():
    candidate, control = _two_tangent_graphs()
    candidate_add = _node(candidate, "aten.add")
    control_add = _node(control, "aten.add")
    candidate_result = _node(candidate, "operator.getitem")["id"]
    control_result = _node(control, "operator.getitem")["id"]
    candidate_second_tangent = _tangent_nodes(candidate)[1]["id"]
    control_second_tangent = _tangent_nodes(control)[1]["id"]
    candidate_add["args"] = {
        "tuple": [{"ref": candidate_result}, {"ref": candidate_second_tangent}]
    }
    control_add["args"] = {
        "tuple": [
            {"ref": _tangent_nodes(control)[0]["id"]},
            {"ref": control_second_tangent},
        ]
    }
    _append_accumulation(candidate, candidate_add["id"], candidate_result)
    _append_accumulation(control, control_add["id"], control_result)

    result = analyze(Capture.from_dict(candidate), Capture.from_dict(control))
    assert result.outcome == Outcome.VIOLATION
    assert result.collapse is not None
    assert result.collapse.replaced_part == "tangent:output:0"
    assert "tangent:output:1" in result.collapse.expected_parts

def test_collapse_never_substitutes_a_different_tangent_role():
    candidate, control = _two_tangent_graphs()
    control_add = _node(control, "aten.add")
    control_add["args"] = {
        "tuple": [
            {"ref": _node(control, "operator.getitem")["id"]},
            {"ref": _tangent_nodes(control)[1]["id"]},
        ]
    }

    result = analyze(Capture.from_dict(candidate), Capture.from_dict(control))
    assert result.outcome == Outcome.UNKNOWN
    assert "not an exact tangent-to-result substitution" in result.unknown_detail

def test_collapse_rejects_duplicate_replaced_tangent_contributions():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    candidate_raw = candidate.to_dict()
    control_raw = control.to_dict()
    candidate_add = _node(candidate_raw, "aten.add")
    control_add = _node(control_raw, "aten.add")
    candidate_tangent = _tangent_node(candidate_raw)["id"]
    control_tangent = _tangent_node(control_raw)["id"]
    candidate_result = _node(candidate_raw, "operator.getitem")["id"]
    candidate_add["args"] = {
        "tuple": [{"ref": candidate_tangent}, {"ref": candidate_result}]
    }
    control_add["args"] = {
        "tuple": [{"ref": control_tangent}, {"ref": control_tangent}]
    }

    result = analyze(Capture.from_dict(candidate_raw), Capture.from_dict(control_raw))

    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.TWIN_RELATION,)
    assert result.collapse is None
    assert "repeats tangent:output:0" in result.unknown_detail

def test_duplicate_tangent_output_roles_fail_closed_while_the_canonical_pair_collapses():
    canonical_candidate, canonical_control = _micrograph(True, "direct", False, 1, False)
    canonical = analyze(canonical_candidate, canonical_control)
    assert canonical.outcome == Outcome.VIOLATION
    assert canonical.collapse is not None

    duplicate_candidate, duplicate_control = _two_tangent_graphs()
    for raw in (duplicate_candidate, duplicate_control):
        tangents = _tangent_nodes(raw)
        tangents[1]["role"] = copy.deepcopy(tangents[0]["role"])

    result = analyze(
        Capture.from_dict(duplicate_candidate),
        Capture.from_dict(duplicate_control),
    )

    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.GRAPH_EFFECT,)
    assert "duplicate tangent output role tangent:output:0" in result.unknown_detail

def test_collapse_requires_a_marked_gradient_accumulation_producer_in_both_lanes():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    candidate_raw = candidate.to_dict()
    control_raw = control.to_dict()
    for raw in (candidate_raw, control_raw):
        _node(raw, "aten.add")["gradient_accumulation"] = False

    result = analyze(
        Capture.from_dict(candidate_raw),
        Capture.from_dict(control_raw),
    )

    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.TWIN_RELATION,)
    assert "marked gradient-accumulation add" in result.unknown_detail


@pytest.mark.parametrize("target", ["aten.view", "aten.detach", "aten.clone", "aten.mul"])
def test_collapse_does_not_cross_an_operation_after_the_marked_add(target):
    candidate, control = _micrograph(True, "direct", False, 1, False)
    lanes = []
    for capture in (candidate, control):
        raw = capture.to_dict()
        nodes = raw["backward"]["nodes"]
        output = nodes.pop()
        accumulation = _node(raw, "aten.add")
        wrapped = copy.deepcopy(accumulation)
        args = [{"ref": accumulation["id"]}]
        if target == "aten.view":
            args.append(list(accumulation["tensor"]["shape"]))
        elif target == "aten.mul":
            args.append(1)
        wrapped.update(
            id=f"n{len(nodes)}", target=target,
            args={"tuple": args}, gradient_accumulation=False,
        )
        if target in {"aten.clone", "aten.mul"}:
            wrapped["tensor"]["alias_group"] = "post-accumulation"
        output["id"] = f"n{len(nodes) + 1}"
        output["args"]["tuple"][0]["tuple"][0]["ref"] = wrapped["id"]
        nodes.extend([wrapped, output])
        lanes.append(Capture.from_dict(raw))

    result = analyze(*lanes)

    assert result.outcome is Outcome.UNKNOWN
    assert result.reasons == (Reason.TWIN_RELATION,)
    assert result.collapse is None
