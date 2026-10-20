from __future__ import annotations

import copy

import pytest

from daggertrace.model import Capture
from daggertrace.replay import ReplayError, replay_certificate
from daggertrace.replay_graph import _fresh_stride, _graph
from daggertrace.replay_v2 import _capture, _execute
from daggertrace.validation_v2_fixtures import _micrograph, _unsupported_kernel_capture
from daggertrace.version import CERTIFICATE_SCHEMA
from _ownership_fixtures import _with_explicit_saved_output
from _replay_fixtures import (
    _canonical_storage_root_case,
    _replayable_micrograph,
    _sync_hop_getitems,
    _write_canonical,
)


def _v2_replayable_capture(**kwargs):
    options = {
        "alias_destination": False,
        "contribution": "none",
        "reverse_operands": False,
        "view_depth": 0,
        "noise": False,
    }
    options.update(kwargs)
    candidate, _ = _micrograph(**options)
    return _replayable_micrograph(candidate)


def test_replay_v2_capture_accepts_source_within_ast_bounds():
    capture, _ = _micrograph(False, "none", False, 0, False)

    assert _replayable_micrograph(capture)["schema"] == "daggertrace.aot-capture/2"


@pytest.mark.parametrize(
    ("extension_kind", "bound_message"),
    [
        pytest.param("depth", "target source AST exceeds its depth bound", id="depth"),
        pytest.param(
            "statements",
            "target source AST exceeds its statement bound",
            id="statements",
        ),
        pytest.param("nodes", "target source AST exceeds its node bound", id="nodes"),
    ],
)
def test_replay_v2_capture_applies_ast_bounds_before_effect_walk(
    extension_kind, bound_message
):
    capture, _ = _micrograph(False, "none", False, 0, False)
    extensions = {
        "depth": "0" + " + 0" * 140,
        "statements": "\n    ".join(
            f"value_{index} = 0" for index in range(4097)
        ),
        "nodes": "[" + ", ".join("0 + 0" for _ in range(4500)) + "]",
    }
    mutated = _unsupported_kernel_capture(capture, extensions[extension_kind])

    with pytest.raises(ReplayError, match=bound_message) as raised:
        _replayable_micrograph(mutated)

    assert str(raised.value) == bound_message
    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value.__cause__) == bound_message


@pytest.mark.parametrize("target", ["aten.zeros_like", "aten.clone"])
@pytest.mark.parametrize("alias_group", [None, "tangent"])
def test_replay_rejects_fresh_result_without_new_captured_storage_identity(
    target, alias_group
):
    _, control = _micrograph(True, "direct", False, 1, False)
    _execute(control.to_dict())
    malicious = copy.deepcopy(control.to_dict())
    allocation = next(
        node for node in malicious["backward"]["nodes"] if node["target"] == "aten.zeros_like"
    )
    allocation["target"] = target
    allocation["tensor"]["alias_group"] = alias_group

    with pytest.raises(ReplayError, match="fresh result"):
        _execute(malicious)


def test_replay_allocation_grammar_rejects_extra_arguments_tensor_refs_and_unknown_kwargs():
    capture, _ = _micrograph(False, "none", False, 1, True)

    edits = (
        lambda raw: next(
            node for node in raw["backward"]["nodes"] if node["target"] == "aten.empty"
        )["args"]["tuple"].append([1]),
        lambda raw: next(
            node for node in raw["backward"]["nodes"] if node["target"] == "aten.zeros_like"
        )["args"].update(tuple=[[2, 2]]),
        lambda raw: next(
            node for node in raw["backward"]["nodes"] if node["target"] == "aten.empty"
        )["args"].update(tuple=[{"ref": "n0"}]),
        lambda raw: next(
            node for node in raw["backward"]["nodes"] if node["target"] == "aten.zeros_like"
        )["kwargs"].update(unexpected=True),
        lambda raw: next(
            node for node in raw["backward"]["nodes"] if node["target"] == "aten.zeros_like"
        ).update(args=[{"ref": "n1"}]),
    )

    for edit in edits:
        raw = copy.deepcopy(capture.to_dict())
        edit(raw)
        with pytest.raises(ReplayError):
            _execute(raw)


def test_replay_accepts_only_fresh_preserving_allocation_keyword_fields():
    capture, _ = _micrograph(False, "none", False, 1, False)
    raw = capture.to_dict()
    allocation = next(
        node for node in raw["backward"]["nodes"] if node["target"] == "aten.zeros_like"
    )
    allocation["kwargs"] = {
        "dtype": {"torch": "float32"},
        "layout": {"torch": "strided"},
        "device": "cuda:0",
        "pin_memory": False,
        "memory_format": {"torch": "contiguous_format"},
    }
    allocation["tensor"]["device"] = "cuda:0"
    _sync_hop_getitems(raw)

    assert _execute(raw)["borrowed"] == []


@pytest.mark.parametrize(
    ("target", "args"),
    [
        ("aten.empty_like", {"tuple": [{"ref": "n1"}]}),
        ("aten.zeros", {"tuple": [[2, 2]]}),
    ],
)
def test_replay_accepts_fresh_empty_like_and_zeros_allocations(target, args):
    capture, _ = _micrograph(False, "none", False, 1, False)
    raw = capture.to_dict()
    allocation = next(
        node for node in raw["backward"]["nodes"] if node["target"] == "aten.zeros_like"
    )
    allocation["target"] = target
    allocation["args"] = args
    if target == "aten.zeros":
        allocation["kwargs"] = {
            "dtype": {"torch": "float32"},
            "device": "accelerator",
        }

    assert _execute(raw)["borrowed"] == []


@pytest.mark.parametrize(
    ("description", "edit", "options", "match"),
    [
        (
            "add arity",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.add")["args"]["tuple"].pop(),
            {"contribution": "direct"},
            "add grammar",
        ),
        (
            "add literal",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.add")["args"].update(tuple=[{"ref": "n0"}, None]),
            {"contribution": "direct"},
            "add grammar",
        ),
        (
            "mul literal",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.mul")["args"].update(tuple=[{"ref": "n0"}, None]),
            {"contribution": "materialized"},
            "mul grammar",
        ),
        (
            "mul without tensor input",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.mul")["args"].update(tuple=[0.25, 0.5]),
            {"contribution": "materialized"},
            "mul grammar",
        ),
        (
            "view arity",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.view")["args"].update(tuple=[{"ref": "n0"}]),
            {"view_depth": 1},
            "view grammar",
        ),
        (
            "view boolean shape",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.view")["args"].update(tuple=[{"ref": "n0"}, [True]]),
            {"view_depth": 1},
            "view grammar",
        ),
        (
            "sum dimension range",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.mul").update(
                target="aten.sum", args={"tuple": [{"ref": "n0"}, [2]]}
            ),
            {"contribution": "materialized"},
            "sum grammar",
        ),
        (
            "sum dimension type",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.mul").update(
                target="aten.sum", args={"tuple": [{"ref": "n0"}, [True]]}
            ),
            {"contribution": "materialized"},
            "dimension|sum grammar",
        ),
        (
            "empty negative size",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.empty")["args"].update(tuple=[[-1]]),
            {"allocation": "empty"},
            "size",
        ),
        (
            "empty boolean size",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.empty")["args"].update(tuple=[[True]]),
            {"allocation": "empty"},
            "size",
        ),
        (
            "empty float size",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.empty")["args"].update(tuple=[[1.0]]),
            {"allocation": "empty"},
            "size",
        ),
        (
            "empty nested size",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.empty")["args"].update(tuple=[[[1]]]),
            {"allocation": "empty"},
            "size",
        ),
        (
            "empty string size",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.empty")["args"].update(tuple=[["1"]]),
            {"allocation": "empty"},
            "size",
        ),
        (
            "empty oversized size",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.empty")["args"].update(tuple=[[(1 << 63)]]),
            {"allocation": "empty"},
            "size",
        ),
        (
            "allocation dtype literal",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.empty")["kwargs"].update(dtype=[]),
            {"allocation": "empty"},
            "allocation keyword dtype",
        ),
        (
            "allocation device literal",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.empty")["kwargs"].update(device=1),
            {"allocation": "empty"},
            "allocation keyword device",
        ),
        (
            "allocation device range",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.empty")["kwargs"].update(device={"torch": "cuda:65536"}),
            {"allocation": "empty"},
            "allocation keyword device",
        ),
        (
            "allocation pin-memory literal",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.empty")["kwargs"].update(pin_memory=1),
            {"allocation": "empty"},
            "allocation keyword pin_memory",
        ),
        (
            "getitem arity",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "operator.getitem")["args"]["tuple"].pop(),
            {},
            "getitem grammar",
        ),
        (
            "getitem key literal",
            lambda raw: next(node for node in raw["backward"]["nodes"] if node["target"] == "operator.getitem")["args"]["tuple"].__setitem__(1, 1),
            {},
            "getitem grammar",
        ),
    ],
    ids=lambda item: item[0] if isinstance(item, tuple) else None,
)
def test_replay_v2_capture_rejects_operation_grammar_mutations(
    description, edit, options, match
):
    raw = _v2_replayable_capture(**options)
    edit(raw)

    with pytest.raises(ReplayError, match=match):
        _capture(raw, "candidate")


def test_replay_v2_graph_outputs_must_reference_tensor_nodes():
    raw = _v2_replayable_capture()
    forward = raw["forward"]
    allocation = copy.deepcopy(
        next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.zeros_like")
    )
    allocation["id"] = "n1"
    hop = copy.deepcopy(
        next(node for node in raw["backward"]["nodes"] if node["target"] == "triton.functional")
    )
    hop["id"] = "n2"
    output = copy.deepcopy(forward["nodes"][-1])
    output["id"] = "n3"
    forward["nodes"] = [
        copy.deepcopy(forward["nodes"][0]),
        allocation,
        hop,
        output,
    ]
    forward["output_roles"] = [{"kind": "output", "index": 0}]

    output["args"] = {"tuple": [{"tuple": [{"ref": "n1"}]}]}
    _graph(forward, "candidate.forward", "forward")

    allocation["target"] = "aten.empty"
    allocation["args"] = {"tuple": [[-1]]}
    allocation["kwargs"] = {
        "dtype": {"torch": "float32"},
        "device": "accelerator",
    }
    with pytest.raises(ReplayError, match="size"):
        _graph(forward, "candidate.forward", "forward")
    allocation["target"] = "aten.zeros_like"
    allocation["args"] = {"tuple": [{"ref": "n0"}]}

    output["args"] = {"tuple": [{"tuple": [{"ref": "n2"}]}]}
    with pytest.raises(ReplayError, match="output references a non-tensor node"):
        _graph(forward, "candidate.forward", "forward")


def test_replay_admits_an_explicit_saved_output_placeholder_in_its_domain():
    raw = _with_explicit_saved_output(_v2_replayable_capture())

    captured = _capture(raw, "candidate")
    result = _execute(captured)

    assert result["borrowed"] == []


def test_replay_rejects_an_explicit_saved_output_outside_its_domain():
    raw = _with_explicit_saved_output(_v2_replayable_capture())
    saved = next(
        node
        for node in raw["backward"]["nodes"]
        if node["role"] and node["role"].get("kind") == "saved_output"
    )
    saved["role"]["index"] = 0

    with pytest.raises(ReplayError, match="outside the forward domain"):
        _capture(raw, "candidate")


def test_replay_rejects_a_saved_output_without_a_saved_declaration():
    raw = _with_explicit_saved_output(_v2_replayable_capture())
    raw["forward"]["output_roles"][1] = {"kind": "output", "index": 1}

    with pytest.raises(ReplayError, match="outside the forward domain"):
        _capture(raw, "candidate")


def test_replay_keeps_saved_output_role_families_closed():
    raw = _with_explicit_saved_output(_v2_replayable_capture())
    raw["forward"]["nodes"][0]["role"] = {
        "kind": "saved_output",
        "index": 1,
    }
    with pytest.raises(ReplayError, match="placeholder role changed"):
        _graph(raw["forward"], "candidate.forward", "forward")

    raw = _with_explicit_saved_output(_v2_replayable_capture())
    raw["backward"]["nodes"][0]["role"] = {
        "kind": "tangent",
        "output": {"kind": "saved_output", "index": 1},
    }
    with pytest.raises(ReplayError, match="does not identify a forward output"):
        _capture(raw, "candidate")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shape", [1, 4]),
        ("stride", [1, 2]),
        ("dtype", "float16"),
        ("device", "other"),
        ("storage_offset", 1),
    ],
)
def test_replay_v2_view_rejects_result_metadata_changes(field, value):
    raw = _v2_replayable_capture(view_depth=1)
    view = next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.view")
    view["tensor"][field] = value

    with pytest.raises(ReplayError, match="view result"):
        _capture(raw, "candidate")


def test_replay_v2_accepts_an_alias_with_complete_source_metadata():
    raw = _v2_replayable_capture(view_depth=1)
    alias = next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.view")
    alias["target"] = "aten.alias"
    alias["args"] = {"tuple": [copy.deepcopy(alias["args"]["tuple"][0])]}

    assert _capture(raw, "candidate")["schema"] == "daggertrace.aot-capture/2"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shape", [1, 4]),
        ("stride", [1, 2]),
        ("dtype", "float16"),
        ("device", "other"),
        ("storage_offset", 1),
    ],
)
def test_replay_v2_alias_rejects_complete_metadata_changes(field, value):
    raw = _v2_replayable_capture(view_depth=1)
    alias = next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.view")
    alias["target"] = "aten.alias"
    alias["args"] = {"tuple": [copy.deepcopy(alias["args"]["tuple"][0])]}
    alias["tensor"][field] = value

    with pytest.raises(ReplayError, match="alias layout"):
        _capture(raw, "candidate")


@pytest.mark.parametrize("target", ["aten.empty_like", "aten.zeros_like", "aten.clone"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shape", [1, 4]),
        ("dtype", "float16"),
        ("device", "other"),
    ],
)
def test_replay_v2_like_and_clone_reject_source_metadata_changes(target, field, value):
    raw = _v2_replayable_capture()
    node = next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.zeros_like")
    node["target"] = target
    node["tensor"][field] = value

    with pytest.raises(ReplayError, match="like result metadata|clone layout"):
        _capture(raw, "candidate")


@pytest.mark.parametrize("target", ["aten.empty_like", "aten.zeros_like", "aten.clone"])
@pytest.mark.parametrize(
    ("field", "value"),
    [("stride", [1, 2]), ("storage_offset", 1)],
)
def test_replay_v2_like_and_clone_reject_impossible_fresh_layout(target, field, value):
    raw = _v2_replayable_capture(view_depth=1)
    node = next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.zeros_like")
    node["target"] = target
    node["tensor"][field] = value

    with pytest.raises(ReplayError, match="layout"):
        _capture(raw, "candidate")


@pytest.mark.parametrize("target", ["aten.empty_like", "aten.zeros_like"])
def test_replay_v2_like_accepts_dtype_device_and_memory_format_overrides(target):
    raw = _v2_replayable_capture()
    node = next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.zeros_like")
    node["target"] = target
    node["kwargs"] = {
        "dtype": {"torch": "float16"},
        "device": "cuda:1",
        "memory_format": {"torch": "contiguous_format"},
    }
    node["tensor"].update(dtype="float16", device="cuda:1")
    _sync_hop_getitems(raw)

    assert _capture(raw, "candidate")["schema"] == "daggertrace.aot-capture/2"


@pytest.mark.parametrize("target", ["aten.empty_like", "aten.zeros_like"])
@pytest.mark.parametrize(
    ("key", "value"),
    [("dtype", {"torch": "float16"}), ("device", "cuda:1")],
)
def test_replay_v2_like_rejects_result_override_mismatch(target, key, value):
    raw = _v2_replayable_capture()
    node = next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.zeros_like")
    node["target"] = target
    node["kwargs"] = {key: value}

    with pytest.raises(ReplayError, match="like result metadata"):
        _capture(raw, "candidate")


@pytest.mark.parametrize("target", ["aten.empty", "aten.zeros"])
def test_replay_v2_empty_and_zeros_result_shape_matches_requested_size(target):
    raw = _v2_replayable_capture(allocation="empty")
    allocation = next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.empty")
    allocation["target"] = target

    assert _capture(raw, "candidate")["schema"] == "daggertrace.aot-capture/2"

    for edit, match in (
        (lambda node: node["kwargs"].pop("dtype"), "declare dtype and device"),
        (lambda node: node["kwargs"].pop("device"), "declare dtype and device"),
        (lambda node: node["tensor"].update(shape=[2]), "allocation result metadata"),
        (lambda node: node["tensor"].update(dtype="float16"), "allocation result metadata"),
        (lambda node: node["tensor"].update(device="other"), "allocation result metadata"),
        (lambda node: node["tensor"].update(storage_offset=1), "allocation result layout"),
        (lambda node: node["tensor"].update(stride=[1, 2]), "allocation result layout"),
        (lambda node: node["args"].update(tuple=[[(1 << 62) + 1, 2]]), "product"),
    ):
        malformed = copy.deepcopy(raw)
        edit(next(node for node in malformed["backward"]["nodes"] if node["target"] == target))
        with pytest.raises(ReplayError, match=match):
            _capture(malformed, "candidate")


@pytest.mark.parametrize(
    ("shape", "source_shape", "source_stride", "kwargs", "expected"),
    [
        ([2, 2], [2, 2], [1, 2], {}, [1, 2]),
        ([2, 2], [2, 2], [3, 1], {}, [2, 1]),
        ([0, 2], [0, 2], [2, 1], {}, None),
        ([1, 2], [1, 2], [2, 1], {}, None),
        ([2, 2], None, None, {"memory_format": {"torch": "preserve_format"}}, None),
    ],
    ids=["dense", "fallback", "zero", "ambiguous", "source-free-preserve"],
)
def test_replay_fresh_stride_independently_mirrors_the_closed_layout_language(
    shape, source_shape, source_stride, kwargs, expected
):
    source = (
        None
        if source_shape is None
        else {"shape": source_shape, "stride": source_stride}
    )
    if expected is None:
        with pytest.raises(ReplayError, match="layout is not provable"):
            _fresh_stride(source, shape, kwargs, "layout")
    else:
        assert _fresh_stride(source, shape, kwargs, "layout") == expected


@pytest.mark.parametrize(
    ("requested", "result_shape", "result_stride"),
    [
        ([2, 0], [2, 0], [0, 1]),
        ([-1, 2], [0, 2], [2, 1]),
    ],
    ids=["reshape", "ambiguous-inference"],
)
def test_replay_zero_numel_view_reshape_fails_closed(
    requested, result_shape, result_stride
):
    capture = _v2_replayable_capture(view_depth=1)
    tangent = next(
        node
        for node in capture["backward"]["nodes"]
        if (node.get("role") or {}).get("kind") == "tangent"
    )
    view = next(node for node in capture["backward"]["nodes"] if node["target"] == "aten.view")
    tangent["tensor"].update(shape=[0, 2], stride=[2, 1])
    view["args"] = {"tuple": [{"ref": tangent["id"]}, requested]}
    view["tensor"].update(shape=result_shape, stride=result_stride)

    with pytest.raises(ReplayError, match="view result stride"):
        _capture(capture, "candidate")


@pytest.mark.parametrize(
    ("source_shape", "source_stride", "requested", "runtime_stride", "forged_stride"),
    [
        ([1, 2], [0, 1], [1, 2], [2, 1], [0, 1]),
        ([2, 1], [0, 1], [1, 2, 1], [0, 0, 1], [0, 0, 0]),
        ([2, 1, 2], [2, 2, 1], [4], [1], [2]),
    ],
    ids=["identity-singleton", "trailing-singleton", "dense-singleton-source"],
)
def test_replay_requires_exact_singleton_view_strides(
    source_shape, source_stride, requested, runtime_stride, forged_stride
):
    capture = _v2_replayable_capture(view_depth=1)
    tangent = next(
        node
        for node in capture["backward"]["nodes"]
        if (node.get("role") or {}).get("kind") == "tangent"
    )
    view = next(node for node in capture["backward"]["nodes"] if node["target"] == "aten.view")
    allocation = next(
        node for node in capture["backward"]["nodes"] if node["target"] == "aten.zeros_like"
    )
    allocation_stride = [1] * len(requested)
    for index in range(len(requested) - 2, -1, -1):
        allocation_stride[index] = allocation_stride[index + 1] * requested[index + 1]
    tangent["tensor"].update(shape=source_shape, stride=source_stride)
    view["args"] = {"tuple": [{"ref": tangent["id"]}, requested]}
    view["tensor"].update(shape=requested, stride=runtime_stride)
    allocation["kwargs"] = {"memory_format": {"torch": "torch.contiguous_format"}}
    allocation["tensor"].update(shape=requested, stride=allocation_stride)
    _sync_hop_getitems(capture)

    assert _capture(copy.deepcopy(capture), "candidate")

    view["tensor"]["stride"] = forged_stride
    with pytest.raises(ReplayError, match="view result stride"):
        _capture(capture, "candidate")


@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: next(
            node for node in raw["backward"]["nodes"] if node["target"] == "triton.functional"
        )["kwargs"]["grid"].__setitem__(0, {"tuple": [1 << 63, 1, 1]}),
        lambda raw: next(
            node for node in raw["backward"]["nodes"] if node["target"] == "triton.functional"
        )["kwargs"]["grid"].__setitem__(0, {"tuple": [(1 << 62) + 1, 2, 1]}),
    ],
    ids=["i64", "product"],
)
def test_replay_v2_hop_grid_is_bounded(edit):
    raw = _v2_replayable_capture()
    edit(raw)

    with pytest.raises(ReplayError, match="grid"):
        _capture(raw, "candidate")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shape", [1] * 65),
        ("shape", [1 << 63]),
        ("shape", [(1 << 62), 4]),
        ("stride", [-1, 1]),
        ("stride", [1 << 63]),
        ("storage_offset", 1 << 63),
    ],
    ids=["rank", "shape-i64", "shape-product", "negative-stride", "stride-i64", "offset-i64"],
)
def test_replay_v2_tensor_metadata_is_bounded(field, value):
    raw = _v2_replayable_capture()
    tensor = raw["backward"]["nodes"][0]["tensor"]
    tensor[field] = value
    if field == "shape" and len(value) != len(tensor["stride"]):
        tensor["stride"] = [1] * len(value)

    with pytest.raises(ReplayError):
        _capture(raw, "candidate")


@pytest.mark.parametrize("target", ["operator.getitem", "aten.mul", "aten.sum", "aten.add"])
def test_replay_v2_functional_result_metadata_is_exact(target):
    capture, node = _canonical_storage_root_case(target)
    node_id = node["id"]
    capture = _replayable_micrograph(Capture.from_dict(capture))
    node = next(node for node in capture["backward"]["nodes"] if node["id"] == node_id)
    assert _capture(capture, "candidate")["schema"] == "daggertrace.aot-capture/2"

    node["tensor"]["shape"] = [1, 4]
    with pytest.raises(ReplayError, match="result metadata"):
        _capture(capture, "candidate")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stride", [1, 2]),
        ("dtype", "float16"),
        ("device", "other"),
        ("storage_offset", 1),
        ("alias_group", "destination"),
        ("alias_group", "tangent"),
    ],
)
def test_replay_v2_getitem_result_has_fresh_storage_and_selected_write_metadata(field, value):
    capture = _v2_replayable_capture()
    result = next(node for node in capture["backward"]["nodes"] if node["target"] == "operator.getitem")

    result["tensor"][field] = value
    with pytest.raises(ReplayError, match="result metadata|fresh result"):
        _capture(capture, "candidate")


def test_replay_rejects_v2_out_of_signed_i64_before_certificate_shape(tmp_path):
    path = tmp_path / "certificate.json"
    _write_canonical(
        path,
        {
            "schema": CERTIFICATE_SCHEMA,
            "target": {"specialization": {"value": 1 << 63}},
        },
    )

    with pytest.raises(ReplayError, match="outside signed-64"):
        replay_certificate(path)


def test_replay_v2_rejects_a_duplicate_hop_output_selection():
    raw = _v2_replayable_capture(view_depth=1)
    nodes = raw["backward"]["nodes"]
    output = nodes.pop()
    duplicate = copy.deepcopy(next(node for node in nodes if node["target"] == "operator.getitem"))
    duplicate["id"] = "n5"
    output["id"] = "n6"
    nodes.extend([duplicate, output])

    with pytest.raises(ReplayError, match="duplicates a HOP output selection"):
        _capture(raw, "candidate")
