from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from daggertrace.errors import InvalidArtifact, UnsupportedCapture
from daggertrace.model import Capture, Outcome, Reason
from daggertrace.ownership import analyze
from daggertrace.ownership_v2_validate import _validate_v2_runtime_tensor
from daggertrace.replay import ReplayError
from daggertrace.replay_graph import _runtime_tensor
from daggertrace.validation_v2_fixtures import _micrograph, _unsupported_kernel_capture
from _ownership_fixtures import (
    _edited,
    _make_v2_alias,
    _node,
    _output_node,
    _sync_hop_getitems,
    _tangent_node,
    _two_input_raw,
    _v2_functional_result_capture,
)


@pytest.mark.parametrize(("overrides", "accepted"), [
    ({"stride": [-1, 1]}, False),
    ({"stride": [1 << 63, 1]}, False),
    ({"stride": [True, 1]}, False),
    ({"stride": [0, 1]}, True),
    ({"stride": [(1 << 63) - 1, 1]}, True),
    ({"shape": [1 << 62, 4]}, False),
    ({"shape": [0, 2]}, True),
    ({"shape": [1 << 63, 2]}, False),
    ({"storage_offset": 1 << 63}, False),
    ({"storage_offset": 0}, True),
    ({"shape": [], "stride": []}, True),
    ({"shape": [1] * 64, "stride": [1] * 64}, True),
    ({"shape": [1] * 65, "stride": [1] * 65}, False),
])
def test_runtime_diagnostic_numeric_domain_matches_independent_replay(overrides, accepted):
    tensor = {
        "shape": [2, 2], "stride": [2, 1], "storage_offset": 0,
        "dtype": "float32", "device": "cuda:0", "requires_grad": True,
    }
    tensor.update(overrides)

    for validate, error in (
        (_validate_v2_runtime_tensor, UnsupportedCapture),
        (_runtime_tensor, ReplayError),
    ):
        if accepted:
            validate(tensor, "diagnostic")
        else:
            with pytest.raises(error):
                validate(tensor, "diagnostic")


def test_v2_nonfinite_constants_raise_a_normalized_admission_error():
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    hop = next(node for node in candidate.backward.nodes if node.target == "triton.functional")
    hop.kwargs["constants"] = {"BLOCK_SIZE": float("nan")}

    with pytest.raises(InvalidArtifact, match="canonically digested"):
        analyze(candidate)

def test_v2_analyzer_uses_capture_admission_for_direct_models():
    candidate, _ = _micrograph(False, "direct", False, 1, False)

    result = analyze(replace(candidate, label=""))

    assert result.outcome is Outcome.UNKNOWN
    assert result.unknown_detail == "capture.label must be nonempty"


def test_v2_analyzer_rejects_malformed_exact_candidate_and_control():
    candidate, control = _micrograph(False, "direct", False, 1, False)
    candidate_nodes = (
        *candidate.backward.nodes[:-1],
        replace(candidate.backward.nodes[-1], id="n999"),
    )
    malformed_candidate = replace(
        candidate,
        backward=replace(candidate.backward, nodes=candidate_nodes),
    )
    malformed_control = replace(
        control,
        kernels=(control.kernels[0], control.kernels[0]),
    )

    assert analyze(malformed_candidate).outcome is Outcome.UNKNOWN
    assert analyze(candidate, malformed_control).outcome is Outcome.UNKNOWN


def test_v2_reason_classifier_does_not_scan_unsupported_target_names():
    candidate, _ = _micrograph(False, "direct", False, 1, False)

    result = analyze(
        _edited(
            candidate,
            lambda raw: _node(raw, "aten.zeros_like").update(
                target="unsupported:MEMORY_MODEL"
            ),
        )
    )

    assert result.outcome is Outcome.UNKNOWN
    assert result.reasons == (Reason.GRAPH_EFFECT,)
    assert result.unknown_detail == (
        "unsupported captured targets: ['unsupported:MEMORY_MODEL']"
    )


@pytest.mark.parametrize(
    ("expression", "reason", "detail"),
    [
        pytest.param(
            "MEMORY_MODEL(read_role_0)",
            Reason.SOURCE_PROFILE,
            "call MEMORY_MODEL is outside the supported source subset",
            id="unknown-call-containing-reason-token",
        ),
        pytest.param(
            "tl.atomic_cas(write_role_0, 0, 0)",
            Reason.MEMORY_MODEL,
            "MEMORY_MODEL: Triton operation tl.atomic_cas",
            id="owner-emitted-memory-protocol-prefix",
        ),
    ],
)
def test_v2_source_reason_requires_an_owner_prefix(expression, reason, detail):
    candidate, _ = _micrograph(False, "direct", False, 1, False)

    result = analyze(_unsupported_kernel_capture(candidate, expression))

    assert result.outcome is Outcome.UNKNOWN
    assert result.reasons == (reason,)
    assert detail in result.unknown_detail


@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: _node(raw, "aten.view")["args"]["tuple"].append(1),
        lambda raw: _node(raw, "aten.view")["args"].update(tuple=[{"ref": "n0"}]),
        lambda raw: _node(raw, "aten.view")["tensor"].update(dtype="float16"),
        lambda raw: _node(raw, "aten.zeros_like").update(
            target="aten.clone", args={"tuple": [{"ref": "n1"}, {"ref": "n1"}]}
        ),
    ],
)
def test_v2_analyzer_rejects_alias_and_clone_contract_drift(edit):
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    raw = copy.deepcopy(candidate.to_dict())
    edit(raw)

    result = analyze(Capture.from_dict(raw))

    assert result.outcome is Outcome.UNKNOWN

@pytest.mark.parametrize(
    ("edit", "detail"),
    [
        (
            lambda raw: raw["diagnostics"]["inputs"].append({"requires_grad": False}),
            "capture diagnostics inputs are malformed",
        ),
        (
            lambda raw: raw["diagnostics"]["inputs"].pop(),
            "capture diagnostics inputs are malformed",
        ),
        (
            lambda raw: raw["diagnostics"]["inputs"][0].update(requires_grad=1),
            "capture diagnostics requires_grad must be boolean",
        ),
    ],
)
def test_v2_diagnostics_inputs_must_match_primal_domain_with_boolean_grad_flags(edit, detail):
    raw = _two_input_raw()
    edit(raw)

    result = analyze(Capture.from_dict(raw))

    assert result.outcome is Outcome.UNKNOWN
    assert result.unknown_detail == detail

@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: raw["diagnostics"].pop("grad_output"),
        lambda raw: raw["diagnostics"].update(extra=None),
        lambda raw: raw["diagnostics"]["inputs"][0].pop("shape"),
        lambda raw: raw["diagnostics"]["inputs"][0].update(shape=[True, 2]),
        lambda raw: raw["diagnostics"]["grad_output"].pop("dtype"),
        lambda raw: raw["diagnostics"]["grad_output"].update(storage_offset=-1),
    ],
)
def test_v2_diagnostics_are_closed_runtime_records(edit):
    raw = _two_input_raw()
    assert analyze(Capture.from_dict(raw)).outcome is Outcome.SAFE_IN_FRAGMENT
    edit(raw)

    assert analyze(Capture.from_dict(raw)).outcome is Outcome.UNKNOWN

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shape", [-1, 2]),
        ("dtype", ""),
        ("device", ""),
        ("storage_offset", -1),
        ("alias_group", ""),
    ],
)
def test_invalid_tensor_metadata_is_rejected_before_ownership_outcome(field, value):
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    raw = copy.deepcopy(candidate.to_dict())
    _tangent_node(raw)["tensor"][field] = value

    try:
        malformed = Capture.from_dict(raw)
    except InvalidArtifact:
        return

    result = analyze(malformed)
    assert result.outcome == Outcome.UNKNOWN
    assert result.outcome not in {Outcome.SAFE_IN_FRAGMENT, Outcome.VIOLATION}

@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: raw["framework"].pop("python"),
        lambda raw: raw["framework"].update(extra="not-admitted"),
        lambda raw: raw["framework"].update(python=""),
        lambda raw: raw["framework"].update(cuda_runtime=None),
        lambda raw: raw["framework"].update(torch="not-pinned"),
    ],
)
def test_framework_profile_is_closed_and_generic_environment_fields_are_typed(edit):
    candidate, _ = _micrograph(False, "direct", False, 1, False)

    result = analyze(_edited(candidate, edit))

    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.FRAMEWORK_PROFILE,)

def test_framework_profile_does_not_pin_generic_python_or_cuda_values():
    candidate, _ = _micrograph(False, "direct", False, 1, False)

    result = analyze(
        _edited(
            candidate,
            lambda raw: raw["framework"].update(python="future", cuda_runtime="future-cuda"),
        )
    )

    assert result.outcome == Outcome.SAFE_IN_FRAGMENT

def test_view_metadata_must_preserve_referenced_storage_identity():
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    assert analyze(candidate).outcome == Outcome.VIOLATION

    result = analyze(
        _edited(candidate, lambda raw: _node(raw, "aten.view")["tensor"].update(alias_group="other"))
    )
    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.GRAPH_EFFECT,)
    assert "view node" in result.unknown_detail

@pytest.mark.parametrize(
    ("field", "value", "reason", "detail"),
    [
        ("shape", [1, 4], Reason.GRAPH_EFFECT, "result metadata"),
        ("stride", [1, 2], Reason.GRAPH_EFFECT, "result metadata"),
        ("dtype", "float16", Reason.GRAPH_EFFECT, "result metadata"),
        ("device", "other", Reason.GRAPH_EFFECT, "result metadata"),
        ("storage_offset", 1, Reason.GRAPH_EFFECT, "result metadata"),
        ("alias_group", "destination", Reason.REGION_ALIAS, "fresh result"),
        ("alias_group", "tangent", Reason.REGION_ALIAS, "fresh result"),
    ],
)
def test_hop_getitem_result_has_fresh_storage_and_selected_write_metadata(field, value, reason, detail):
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    assert analyze(candidate).outcome == Outcome.SAFE_IN_FRAGMENT

    result = analyze(
        _edited(
            candidate,
            lambda raw: _node(raw, "operator.getitem")["tensor"].update(**{field: value}),
        )
    )
    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (reason,)
    assert detail in result.unknown_detail

def test_getitem_rejects_extra_arguments_and_keyword_arguments():
    candidate, _ = _micrograph(True, "direct", False, 1, False)

    def extra_argument(raw):
        _node(raw, "operator.getitem")["args"]["tuple"].append("ignored")

    def keyword_argument(raw):
        _node(raw, "operator.getitem")["kwargs"] = {"ignored": True}

    def list_encoded_arguments(raw):
        getitem = _node(raw, "operator.getitem")
        getitem["args"] = getitem["args"]["tuple"]

    for edit in (extra_argument, keyword_argument, list_encoded_arguments):
        result = analyze(_edited(candidate, edit))
        assert result.outcome == Outcome.UNKNOWN
        assert "grammar" in result.unknown_detail or "tuple structure" in result.unknown_detail

def test_pure_and_add_nodes_require_canonical_positional_grammar():
    candidate, _ = _micrograph(True, "materialized", False, 1, False)

    def extra_pure_argument(raw):
        _node(raw, "aten.mul")["args"]["tuple"].append(1.0)

    def pure_keyword_argument(raw):
        _node(raw, "aten.mul")["kwargs"] = {"alpha": 1.0}

    def list_encoded_pure_arguments(raw):
        pure = _node(raw, "aten.mul")
        pure["args"] = pure["args"]["tuple"]

    def extra_add_argument(raw):
        _node(raw, "aten.add")["args"]["tuple"].append(1.0)

    def add_keyword_argument(raw):
        _node(raw, "aten.add")["kwargs"] = {"alpha": 1.0}

    def list_encoded_add_arguments(raw):
        add = _node(raw, "aten.add")
        add["args"] = add["args"]["tuple"]

    for edit in (
        extra_pure_argument,
        pure_keyword_argument,
        list_encoded_pure_arguments,
        extra_add_argument,
        add_keyword_argument,
        list_encoded_add_arguments,
    ):
        result = analyze(_edited(candidate, edit))
        assert result.outcome == Outcome.UNKNOWN

def test_alias_nodes_require_canonical_positional_grammar_and_empty_kwargs():
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    assert analyze(candidate).outcome == Outcome.VIOLATION

    def keyword_argument(raw):
        _node(raw, "aten.view")["kwargs"] = {"arbitrary": True}

    def list_encoded_arguments(raw):
        alias = _node(raw, "aten.view")
        alias["args"] = alias["args"]["tuple"]

    for edit in (keyword_argument, list_encoded_arguments):
        result = analyze(_edited(candidate, edit))
        assert result.outcome == Outcome.UNKNOWN
        assert "alias node" in result.unknown_detail or "tuple structure" in result.unknown_detail

def test_v2_view_rejects_boolean_shape_but_accepts_integer_shape():
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    assert analyze(candidate).outcome == Outcome.VIOLATION

    malformed = analyze(
        _edited(
            candidate,
            lambda raw: _node(raw, "aten.view")["args"].update(
                tuple=[{"ref": "n0"}, [True]]
            ),
        )
    )

    assert malformed.outcome == Outcome.UNKNOWN
    assert malformed.reasons == (Reason.GRAPH_EFFECT,)

@pytest.mark.parametrize(
    ("requested", "result_shape", "result_stride"),
    [
        ([2, 0], [2, 0], [0, 1]),
        ([-1, 2], [0, 2], [2, 1]),
    ],
    ids=["reshape", "ambiguous-inference"],
)
def test_v2_zero_numel_view_reshape_fails_closed(requested, result_shape, result_stride):
    candidate, _ = _micrograph(False, "none", False, 1, False)
    raw = candidate.to_dict()
    tangent = _tangent_node(raw)
    view = _node(raw, "aten.view")
    tangent["tensor"].update(shape=[0, 2], stride=[2, 1])
    view["args"] = {"tuple": [{"ref": tangent["id"]}, requested]}
    view["tensor"].update(shape=result_shape, stride=result_stride)

    result = analyze(Capture.from_dict(raw))
    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.GRAPH_EFFECT,)
    assert "view node" in result.unknown_detail

@pytest.mark.parametrize(
    ("source_shape", "source_stride", "requested", "runtime_stride", "forged_stride"),
    [
        ([1, 2], [0, 1], [1, 2], [2, 1], [0, 1]),
        ([2, 1], [0, 1], [1, 2, 1], [0, 0, 1], [0, 0, 0]),
        ([2, 1, 2], [2, 2, 1], [4], [1], [2]),
    ],
    ids=["identity-singleton", "trailing-singleton", "dense-singleton-source"],
)
def test_v2_analyzer_requires_exact_singleton_view_strides(
    source_shape, source_stride, requested, runtime_stride, forged_stride
):
    candidate, _ = _micrograph(False, "none", False, 1, False)
    raw = candidate.to_dict()
    tangent = _tangent_node(raw)
    view = _node(raw, "aten.view")
    allocation = _node(raw, "aten.zeros_like")
    allocation_stride = [1] * len(requested)
    for index in range(len(requested) - 2, -1, -1):
        allocation_stride[index] = allocation_stride[index + 1] * requested[index + 1]
    tangent["tensor"].update(shape=source_shape, stride=source_stride)
    view["args"] = {"tuple": [{"ref": tangent["id"]}, requested]}
    view["tensor"].update(shape=requested, stride=runtime_stride)
    allocation["kwargs"] = {"memory_format": {"torch": "torch.contiguous_format"}}
    allocation["tensor"].update(shape=requested, stride=allocation_stride)
    _sync_hop_getitems(raw)

    assert analyze(Capture.from_dict(copy.deepcopy(raw))).outcome == Outcome.SAFE_IN_FRAGMENT

    view["tensor"]["stride"] = forged_stride
    result = analyze(Capture.from_dict(raw))
    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.GRAPH_EFFECT,)
    assert "view node" in result.unknown_detail

def test_v2_mul_rejects_none_literal_but_accepts_numeric_literal():
    candidate, _ = _micrograph(True, "materialized", False, 1, False)
    assert analyze(candidate).outcome == Outcome.VIOLATION

    malformed = analyze(
        _edited(
            candidate,
            lambda raw: _node(raw, "aten.mul")["args"].update(
                tuple=[{"ref": "n0"}, None]
            ),
        )
    )

    assert malformed.outcome == Outcome.UNKNOWN
    assert malformed.reasons == (Reason.GRAPH_EFFECT,)

@pytest.mark.parametrize("target", ["operator.getitem", "aten.mul", "aten.sum", "aten.add"])
def test_v2_functional_results_require_derived_metadata(target):
    raw, node = _v2_functional_result_capture(target)
    node["tensor"]["shape"] = [1, 4]

    result = analyze(Capture.from_dict(raw))
    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.GRAPH_EFFECT,)

@pytest.mark.parametrize("target", ["operator.getitem", "aten.mul", "aten.sum", "aten.add"])
def test_v2_functional_result_metadata_opposites_are_admitted(target):
    raw, _ = _v2_functional_result_capture(target)

    assert analyze(Capture.from_dict(raw)).outcome == Outcome.SAFE_IN_FRAGMENT

def test_v2_analyzer_rejects_duplicate_hop_output_selection():
    candidate, _ = _micrograph(False, "none", False, 1, False)
    nodes = list(candidate.backward.nodes)
    output = nodes.pop()
    duplicate = replace(
        next(node for node in nodes if node.target == "operator.getitem"),
        id="n5",
    )
    duplicate_output = replace(output, id="n6")
    malformed = replace(
        candidate,
        backward=replace(candidate.backward, nodes=tuple([*nodes, duplicate, duplicate_output])),
    )

    result = analyze(malformed)
    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.GRAPH_EFFECT,)
    assert "duplicates a HOP output selection" in result.unknown_detail

def test_v2_alias_requires_source_shape_stride_dtype_device_and_offset():
    fields = (
        ("shape", [1, 4]),
        ("stride", [1, 2]),
        ("dtype", "float16"),
        ("device", "other"),
        ("storage_offset", 1),
    )
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    assert analyze(_edited(candidate, _make_v2_alias)).outcome == Outcome.VIOLATION

    for field, value in fields:
        result = analyze(
            _edited(
                candidate,
                lambda raw, field=field, value=value: (
                    _make_v2_alias(raw),
                    _node(raw, "aten.alias")["tensor"].update(**{field: value}),
                ),
            )
        )
        assert result.outcome == Outcome.UNKNOWN
        assert result.reasons == (Reason.GRAPH_EFFECT,)

@pytest.mark.parametrize("target", ["aten.empty_like", "aten.zeros_like", "aten.clone"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shape", [1, 4]),
        ("dtype", "float16"),
        ("device", "other"),
    ],
)
def test_v2_like_and_clone_reject_source_metadata_changes(target, field, value):
    candidate, _ = _micrograph(False, "none", False, 0, False)

    def edit(raw):
        node = _node(raw, "aten.zeros_like")
        node["target"] = target
        node["tensor"][field] = value

    result = analyze(_edited(candidate, edit))

    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.GRAPH_EFFECT,)

@pytest.mark.parametrize("target", ["aten.empty_like", "aten.zeros_like", "aten.clone"])
@pytest.mark.parametrize(
    ("field", "value"),
    [("stride", [1, 2]), ("storage_offset", 1)],
)
def test_v2_like_and_clone_reject_impossible_fresh_layout(target, field, value):
    candidate, _ = _micrograph(False, "none", False, 1, False)

    def edit(raw):
        node = _node(raw, "aten.zeros_like")
        node["target"] = target
        node["tensor"][field] = value

    result = analyze(_edited(candidate, edit))

    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.GRAPH_EFFECT,)

@pytest.mark.parametrize("target", ["aten.empty_like", "aten.zeros_like", "aten.clone"])
def test_v2_fresh_result_uses_zero_offset_for_a_view_with_nonzero_source_offset(target):
    candidate, _ = _micrograph(False, "none", False, 1, False)

    def edit(raw):
        node = _node(raw, "aten.zeros_like")
        node["target"] = target
        for source in raw["backward"]["nodes"]:
            if source["tensor"] is not None and source["tensor"]["alias_group"] == "tangent":
                source["tensor"]["storage_offset"] = 1

    assert analyze(_edited(candidate, edit)).outcome == Outcome.SAFE_IN_FRAGMENT

@pytest.mark.parametrize("target", ["aten.empty_like", "aten.zeros_like", "aten.clone"])
def test_v2_preserve_format_keeps_a_provably_dense_source_stride(target):
    candidate, _ = _micrograph(False, "none", False, 1, False)

    def edit(raw):
        node = _node(raw, "aten.zeros_like")
        node["target"] = target
        for source in raw["backward"]["nodes"]:
            if source["tensor"] is not None and source["tensor"]["alias_group"] == "tangent":
                source["tensor"]["stride"] = [1, 2]
        node["tensor"]["stride"] = [1, 2]
        _sync_hop_getitems(raw)

    assert analyze(_edited(candidate, edit)).outcome == Outcome.SAFE_IN_FRAGMENT

@pytest.mark.parametrize("target", ["aten.empty_like", "aten.zeros_like"])
def test_v2_like_accepts_dtype_device_and_memory_format_overrides(target):
    candidate, _ = _micrograph(False, "none", False, 0, False)

    def edit(raw):
        node = _node(raw, "aten.zeros_like")
        node["target"] = target
        for source in raw["backward"]["nodes"]:
            if source["tensor"] is not None and source["tensor"]["alias_group"] == "tangent":
                source["tensor"]["stride"] = [1, 2]
        node["kwargs"] = {
            "dtype": {"torch": "float16"},
            "device": "cuda:1",
            "memory_format": {"torch": "contiguous_format"},
        }
        node["tensor"].update(dtype="float16", device="cuda:1", stride=[2, 1])
        _sync_hop_getitems(raw)

    assert analyze(_edited(candidate, edit)).outcome == Outcome.SAFE_IN_FRAGMENT

@pytest.mark.parametrize("target", ["aten.empty_like", "aten.zeros_like"])
@pytest.mark.parametrize(
    ("key", "value"),
    [("dtype", {"torch": "float16"}), ("device", "cuda:1")],
)
def test_v2_like_rejects_result_override_mismatch(target, key, value):
    candidate, _ = _micrograph(False, "none", False, 0, False)

    def edit(raw):
        node = _node(raw, "aten.zeros_like")
        node["target"] = target
        node["kwargs"] = {key: value}

    result = analyze(_edited(candidate, edit))

    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.GRAPH_EFFECT,)

def test_graph_grammar_rejects_malformed_normalized_nodes():
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    assert analyze(candidate).outcome == Outcome.VIOLATION

    edits = (
        lambda raw: _node(raw, "triton.functional").update(args={"tuple": ["ignored"]}),
        lambda raw: _tangent_node(raw).update(kwargs={"ignored": True}),
        lambda raw: _tangent_node(raw).update(target="aten.detach"),
        lambda raw: _node(raw, "aten.view").update(op="made_up"),
        lambda raw: _output_node(raw).update(target="made_up"),
        lambda raw: _output_node(raw).update(kwargs={"ignored": True}),
    )
    for edit in edits:
        result = analyze(_edited(candidate, edit))
        assert result.outcome == Outcome.UNKNOWN
        assert result.reasons == (Reason.GRAPH_EFFECT,)

def test_clone_requires_canonical_positional_grammar_and_empty_kwargs():
    candidate, _ = _micrograph(False, "direct", False, 1, False)

    def make_clone(raw):
        _node(raw, "aten.zeros_like")["target"] = "aten.clone"

    canonical = _edited(candidate, make_clone)
    assert analyze(canonical).outcome == Outcome.SAFE_IN_FRAGMENT

    def keyword_argument(raw):
        make_clone(raw)
        _node(raw, "aten.clone")["kwargs"] = {"arbitrary": True}

    def list_encoded_arguments(raw):
        make_clone(raw)
        clone = _node(raw, "aten.clone")
        clone["args"] = clone["args"]["tuple"]

    for edit in (keyword_argument, list_encoded_arguments):
        result = analyze(_edited(candidate, edit))
        assert result.outcome == Outcome.UNKNOWN
        assert result.reasons == (Reason.GRAPH_EFFECT,)

def test_allocation_grammar_rejects_extra_arguments_tensor_refs_and_unknown_kwargs():
    candidate, _ = _micrograph(False, "none", False, 1, True)

    edits = (
        lambda raw: _node(raw, "aten.empty")["args"]["tuple"].append([1]),
        lambda raw: _node(raw, "aten.zeros_like")["args"].update(
            tuple=[[2, 2]]
        ),
        lambda raw: _node(raw, "aten.empty")["args"].update(
            tuple=[{"ref": _tangent_node(raw)["id"]}]
        ),
        lambda raw: _node(raw, "aten.zeros_like")["kwargs"].update(unexpected=True),
        lambda raw: _node(raw, "aten.zeros_like").update(
            args=[{"ref": _node(raw, "aten.view")["id"]}]
        ),
        lambda raw: _node(raw, "aten.empty")["args"].update(tuple=[[-1]]),
        lambda raw: _node(raw, "aten.empty")["args"].update(tuple=[[True]]),
        lambda raw: _node(raw, "aten.empty")["args"].update(
            tuple=[[(1 << 62) + 1, 2]]
        ),
        lambda raw: _node(raw, "aten.empty")["kwargs"].update(dtype=[]),
        lambda raw: _node(raw, "aten.empty")["kwargs"].update(device={"torch": "cuda:65536"}),
        lambda raw: _node(raw, "aten.empty")["kwargs"].update(pin_memory=1),
        lambda raw: _node(raw, "aten.empty")["tensor"].update(shape=[2], stride=[1]),
    )

    for edit in edits:
        result = analyze(_edited(candidate, edit))
        assert result.outcome == Outcome.UNKNOWN
        assert result.reasons == (Reason.GRAPH_EFFECT,)

    with pytest.raises(InvalidArtifact, match="outside signed-64"):
        _edited(
            candidate,
            lambda raw: _node(raw, "aten.empty")["args"].update(tuple=[[(1 << 63)]]),
        )

def test_allocation_accepts_only_fresh_preserving_keyword_fields():
    candidate, _ = _micrograph(False, "none", False, 1, False)

    result = analyze(
        _edited(
            candidate,
            lambda raw: _node(raw, "aten.zeros_like").update(
                kwargs={
                    "dtype": {"torch": "float32"},
                    "layout": {"torch": "strided"},
                    "device": "accelerator",
                    "pin_memory": False,
                    "memory_format": {"torch": "contiguous_format"},
                }
            ),
        )
    )

    assert result.outcome == Outcome.SAFE_IN_FRAGMENT

@pytest.mark.parametrize("target", ["aten.empty", "aten.zeros"])
def test_allocation_accepts_bounded_size_and_keyword_opposites(target):
    candidate, _ = _micrograph(False, "none", False, 1, True)

    def edit(raw):
        allocation = _node(raw, "aten.empty")
        allocation.update(
            target=target,
            args={"tuple": [[1]]},
            kwargs={
                "dtype": {"torch": "float32"},
                "layout": {"torch": "strided"},
                "device": "accelerator",
                "pin_memory": False,
                "memory_format": {"torch": "contiguous_format"},
            },
        )

    result = analyze(_edited(candidate, edit))

    assert result.outcome == Outcome.SAFE_IN_FRAGMENT

@pytest.mark.parametrize("target", ["aten.empty", "aten.zeros"])
def test_direct_allocations_bind_explicit_dtype_device_and_result_metadata(target):
    candidate, _ = _micrograph(False, "none", False, 1, True)

    for edit in (
        lambda allocation: allocation["kwargs"].pop("dtype"),
        lambda allocation: allocation["kwargs"].pop("device"),
        lambda allocation: allocation["tensor"].update(dtype="float16"),
        lambda allocation: allocation["tensor"].update(device="other"),
    ):
        def mutate(raw, edit=edit):
            allocation = _node(raw, "aten.empty")
            allocation["target"] = target
            edit(allocation)

        result = analyze(_edited(candidate, mutate))
        assert result.outcome == Outcome.UNKNOWN
        assert result.reasons == (Reason.GRAPH_EFFECT,)

@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: _node(raw, "triton.functional")["kwargs"].update(
            grid=[{"tuple": [1 << 63, 1, 1]}]
        ),
        lambda raw: _node(raw, "triton.functional")["kwargs"].update(
            grid=[{"tuple": [(1 << 62) + 1, 2, 1]}]
        ),
        lambda raw: _tangent_node(raw)["tensor"].update(shape=[1] * 65, stride=[1] * 65),
        lambda raw: _tangent_node(raw)["tensor"].update(shape=[1 << 63], stride=[1]),
        lambda raw: _tangent_node(raw)["tensor"].update(shape=[1 << 62, 4], stride=[4, 1]),
        lambda raw: _tangent_node(raw)["tensor"].update(stride=[-1, 1]),
        lambda raw: _tangent_node(raw)["tensor"].update(stride=[1 << 63, 1]),
        lambda raw: _tangent_node(raw)["tensor"].update(storage_offset=1 << 63),
    ],
    ids=["grid-i64", "grid-product", "rank", "shape-i64", "shape-product", "negative-stride", "stride-i64", "offset-i64"],
)
def test_v2_graph_grammar_matches_numeric_bounds(edit):
    candidate, _ = _micrograph(False, "none", False, 1, True)

    try:
        malformed = _edited(candidate, edit)
    except InvalidArtifact:
        return
    result = analyze(malformed)
    assert result.outcome == Outcome.UNKNOWN
    assert result.reasons == (Reason.GRAPH_EFFECT,)

def test_placeholder_tangent_roles_fail_closed_when_malformed_or_out_of_domain():
    candidate, _ = _micrograph(True, "direct", False, 1, False)
    roles = (
        {"kind": "unsupported", "descriptor": "mystery"},
        {"kind": "saved", "descriptor": "mystery"},
        {"kind": "tangent", "output": None},
        {"kind": "tangent", "output": {"kind": "output", "index": True}},
        {"kind": "tangent", "output": {"kind": "output", "index": -1}},
        {"kind": "tangent", "output": {"kind": "output", "index": 1}},
        {
            "kind": "tangent",
            "output": {"kind": "output", "index": 0},
            "unexpected": True,
        },
    )
    for role in roles:
        result = analyze(
            _edited(candidate, lambda raw, role=role: _tangent_node(raw).update(role=role))
        )
        assert result.outcome == Outcome.UNKNOWN
        assert "descriptor" in result.unknown_detail or "output domain" in result.unknown_detail
