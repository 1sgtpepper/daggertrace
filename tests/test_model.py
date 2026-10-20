from __future__ import annotations

import copy
import pickle
from dataclasses import replace
from typing import Any, get_type_hints

import pytest

from daggertrace.canonical import sha256_bytes, sha256_json
from daggertrace import model_core
from daggertrace.errors import InvalidArtifact
from daggertrace.model import (
    Capture,
    Graph,
    Kernel,
    Node,
    TensorMeta,
)
from daggertrace.version import (
    CAPTURE_SCHEMA,
    CERTIFICATE_SCHEMA,
    CHECKER_PROFILE,
    __version__,
)
from daggertrace.validation_v2_fixtures import _micrograph


def test_v04_identity_versions_are_bound_to_the_core_artifacts():
    assert __version__ == "0.4.0"
    assert CAPTURE_SCHEMA == "daggertrace.aot-capture/2"
    assert CERTIFICATE_SCHEMA == "daggertrace.cotangent-ownership/2"
    assert CHECKER_PROFILE == "torch-2.12-triton-hop-whole-storage/2"


def test_model_facade_keeps_serialized_types_at_the_established_import_path():
    tensor = TensorMeta((2, 3), (3, 1), "float32", "cuda:0")

    assert Capture is model_core.Capture
    assert TensorMeta.__module__ == "daggertrace.model"
    assert type(pickle.loads(pickle.dumps(tensor))) is TensorMeta


def test_model_facade_resolves_deferred_public_type_hints():
    assert get_type_hints(Node)["args"] is Any
    assert get_type_hints(Capture)["diagnostics"] == dict[str, Any]


def test_capture_round_trip_is_canonical():
    capture, _ = _micrograph(True, "direct", False, 1, False)
    assert Capture.from_dict(capture) is capture
    assert Capture.from_dict(capture.to_dict()).to_dict() == capture.to_dict()
    assert Capture.from_dict(capture.to_dict()).digest() == capture.digest()


@pytest.mark.parametrize("location", ["node_args", "diagnostics"])
def test_public_capture_dict_recursion_is_invalid_artifact(location):
    from daggertrace.ownership import analyze

    capture, _ = _micrograph(False, "direct", False, 1, False)
    raw = copy.deepcopy(capture.to_dict())
    if location == "node_args":
        node = next(
            node for node in raw["backward"]["nodes"] if node["target"] == "aten.view"
        )
        node["args"]["cycle"] = node["args"]
    else:
        raw["diagnostics"]["cycle"] = raw["diagnostics"]

    for admit in (Capture.from_dict, analyze):
        with pytest.raises(
            InvalidArtifact, match="capture structure is too deep or cyclic"
        ) as error:
            admit(raw)
        assert isinstance(error.value.__cause__, RecursionError)


def test_direct_capture_admission_rejects_non_dense_graph_ids():
    capture, _ = _micrograph(False, "direct", False, 1, False)
    nodes = (*capture.backward.nodes[:-1], replace(capture.backward.nodes[-1], id="n999"))
    malformed = replace(capture, backward=replace(capture.backward, nodes=nodes))

    with pytest.raises(InvalidArtifact, match="dense topological"):
        Capture.from_dict(malformed)


def test_direct_capture_admission_rejects_missing_reference():
    capture, _ = _micrograph(False, "direct", False, 1, False)
    nodes = list(capture.backward.nodes)
    view_index = next(index for index, node in enumerate(nodes) if node.target == "aten.view")
    nodes[view_index] = replace(
        nodes[view_index],
        args={"tuple": [{"ref": "n999"}, [2, 2]]},
    )
    malformed = replace(capture, backward=replace(capture.backward, nodes=tuple(nodes)))

    with pytest.raises(InvalidArtifact, match="non-prior reference"):
        Capture.from_dict(malformed)


def test_direct_capture_admission_rejects_future_reference():
    capture, _ = _micrograph(False, "direct", False, 1, False)
    nodes = list(capture.backward.nodes)
    view_index = next(index for index, node in enumerate(nodes) if node.target == "aten.view")
    later_tensor = next(node for node in nodes[view_index + 1 :] if node.tensor is not None)
    nodes[view_index] = replace(
        nodes[view_index],
        args={"tuple": [{"ref": later_tensor.id}, [2, 2]]},
    )
    malformed = replace(capture, backward=replace(capture.backward, nodes=tuple(nodes)))

    with pytest.raises(InvalidArtifact, match="non-prior reference"):
        Capture.from_dict(malformed)


def test_direct_capture_admission_rejects_duplicate_kernel_ids():
    capture, _ = _micrograph(False, "direct", False, 1, False)
    malformed = replace(capture, kernels=(capture.kernels[0], capture.kernels[0]))

    with pytest.raises(InvalidArtifact, match="kernel ids are not unique"):
        Capture.from_dict(malformed)


@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: next(
            node for node in raw["backward"]["nodes"] if node["target"] == "triton.functional"
        )["kwargs"].update(grid=[]),
        lambda raw: next(
            node for node in raw["backward"]["nodes"] if node["target"] == "triton.functional"
        )["kwargs"].update(grid=[{"tuple": [0, 1, 1]}]),
        lambda raw: raw["backward"].update(output_roles=[]),
        lambda raw: raw["forward"]["output_roles"].extend(
            [{"kind": "saved_output", "index": 0}, {"kind": "saved_output", "index": 0}]
        ),
        lambda raw: next(
            node for node in raw["backward"]["nodes"] if node["target"] == "aten.view"
        )["args"]["tuple"].__setitem__(1, [float("nan"), 2]),
    ],
)
def test_v2_model_admission_rejects_replay_invalid_shapes(edit):
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    raw = copy.deepcopy(candidate.to_dict())
    edit(raw)
    with pytest.raises(InvalidArtifact):
        Capture.from_dict(raw)


def test_v2_capture_admission_accepts_canonical_scalar_boundaries():
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    raw = copy.deepcopy(candidate.to_dict())
    text = "é" * ((1 << 20) // 2)
    _v2_literal_kwargs(raw).update(
        {
            "minimum": -(1 << 63),
            "maximum": (1 << 63) - 1,
            "boolean": True,
            text: text,
        }
    )

    admitted = Capture.from_dict(raw)

    view = next(node for node in admitted.backward.nodes if node.target == "aten.view")
    assert view.kwargs[text] == text


@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: _v2_literal_kwargs(raw).update(value=-(1 << 63) - 1),
        lambda raw: _v2_literal_kwargs(raw).update(value=1 << 63),
        lambda raw: _v2_literal_kwargs(raw).update(value="é" * ((1 << 20) // 2) + "a"),
        lambda raw: _v2_literal_kwargs(raw).update(value="\ud800"),
        lambda raw: _v2_literal_kwargs(raw).update({"\ud800": None}),
        lambda raw: _v2_literal_kwargs(raw).update({"é" * ((1 << 20) // 2) + "a": None}),
        lambda raw: raw.update(label=""),
        lambda raw: raw.update(label="é" * ((1 << 20) // 2) + "a"),
    ],
)
def test_v2_capture_admission_rejects_noncanonical_scalars(edit):
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    raw = copy.deepcopy(candidate.to_dict())
    edit(raw)

    with pytest.raises(InvalidArtifact):
        Capture.from_dict(raw)


def _v2_literal_kwargs(raw):
    return next(node for node in raw["backward"]["nodes"] if node["target"] == "aten.view")["kwargs"]


def test_v2_parser_retains_dict_subclass_compatibility():
    candidate, _ = _micrograph(False, "direct", False, 1, False)

    class CompatibleDict(dict):
        pass

    assert Capture.from_dict(CompatibleDict(candidate.to_dict())).to_dict() == candidate.to_dict()


def test_v2_parser_accepts_schema_string_subclasses():
    candidate, _ = _micrograph(False, "direct", False, 1, False)

    class SchemaString(str):
        def __eq__(self, other):
            raise AssertionError("v2 schema comparison must use the legacy route")

    raw = candidate.to_dict()
    raw["schema"] = SchemaString(CAPTURE_SCHEMA)
    parsed = Capture.from_dict(raw)

    assert parsed.schema == CAPTURE_SCHEMA


def test_v2_graph_rejects_duplicate_hop_output_selection():
    capture, _ = _micrograph(False, "none", False, 1, False)
    raw = capture.to_dict()
    nodes = raw["backward"]["nodes"]
    output = nodes.pop()
    duplicate = copy.deepcopy(next(node for node in nodes if node["target"] == "operator.getitem"))
    duplicate["id"] = "n5"
    output["id"] = "n6"
    nodes.extend([duplicate, output])

    with pytest.raises(InvalidArtifact, match="duplicates a HOP output selection"):
        Capture.from_dict(raw)


def test_capture_rejects_unknown_fields():
    capture, _ = _micrograph(True, "direct", False, 1, False)
    raw = capture.to_dict()
    raw["surprise"] = True
    with pytest.raises(InvalidArtifact, match="fields mismatch"):
        Capture.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shape", [-1, 2]),
        ("shape", [True, 2]),
        ("stride", [-1, 1]),
        ("stride", [1.5, 1]),
        ("dtype", ""),
        ("device", ""),
        ("storage_offset", -1),
        ("alias_group", ""),
    ],
)
def test_tensor_meta_rejects_values_outside_the_replay_boundary(field, value):
    capture, _ = _micrograph(True, "direct", False, 1, False)
    raw = copy.deepcopy(capture.to_dict()["backward"]["nodes"][0]["tensor"])
    raw[field] = value

    with pytest.raises(InvalidArtifact):
        TensorMeta.from_dict(raw)


def test_tensor_meta_rejects_negative_stride_and_allows_zero_stride():
    capture, _ = _micrograph(True, "direct", False, 1, False)
    raw = copy.deepcopy(capture.to_dict()["backward"]["nodes"][0]["tensor"])
    raw["stride"] = [-2, 1]
    raw["alias_group"] = None

    with pytest.raises(InvalidArtifact, match="stride"):
        TensorMeta.from_dict(raw)

    raw["stride"] = [0, 1]
    tensor = TensorMeta.from_dict(raw)
    assert tensor.stride == (0, 1)
    assert tensor.alias_group is None


def test_graph_rejects_forward_reference():
    capture, _ = _micrograph(True, "direct", False, 1, False)
    raw = copy.deepcopy(capture.to_dict())
    raw["backward"]["nodes"][1]["args"] = {"tuple": [{"ref": "n2"}, [2, 2]]}
    with pytest.raises(InvalidArtifact, match="non-prior reference"):
        Capture.from_dict(raw)


def test_kernel_identity_requires_sha256():
    capture, _ = _micrograph(True, "direct", False, 1, False)
    raw = copy.deepcopy(capture.to_dict())
    raw["kernels"][0]["id"] = "not-a-digest"
    with pytest.raises(InvalidArtifact, match="SHA-256"):
        Capture.from_dict(raw)


def test_capture_config_replaces_legacy_guards_field():
    graph = Graph(
        kind="forward",
        nodes=(Node(id="n0", op="output", target="output", args={"tuple": []}, kwargs={}),),
    )
    capture = Capture(
        label="unit",
        framework={},
        capture_config={"fullgraph": True},
        forward=graph,
        backward=Graph(
            kind="backward",
            nodes=(Node(id="n0", op="output", target="output", args={"tuple": []}, kwargs={}),),
        ),
        kernels=(),
    )
    raw = capture.to_dict()
    assert "capture_config" in raw and "guards" not in raw
    raw["guards"] = raw.pop("capture_config")
    with pytest.raises(InvalidArtifact, match="fields mismatch"):
        Capture.from_dict(raw)


def test_node_source_is_not_in_the_schema():
    raw = Node(id="n0", op="output", target="output", args={"tuple": []}, kwargs={}).to_dict()
    assert "source" not in raw
    raw["source"] = "display-only"
    with pytest.raises(InvalidArtifact, match="fields mismatch"):
        Node.from_dict(raw)


def test_kernel_runtime_is_json_and_identity_bound():
    kernel = _kernel()
    assert Kernel.from_dict(kernel.to_dict()) == kernel

    admitted = copy.deepcopy(kernel.to_dict())
    admitted["runtime"]["config"]["num_warps"] = 8
    admitted["id"] = sha256_json(_kernel_identity(admitted))
    assert Kernel.from_dict(admitted).runtime["config"]["num_warps"] == 8

    non_json = copy.deepcopy(kernel.to_dict())
    non_json["runtime"]["config"]["kwargs"] = {"bad": object()}
    with pytest.raises(InvalidArtifact, match="non-JSON value"):
        Kernel.from_dict(non_json)


@pytest.mark.parametrize("num_warps", [1, 2, 4, 8, 16, 32])
def test_kernel_runtime_accepts_bounded_power_of_two_warp_counts(num_warps):
    raw = _kernel().to_dict()
    raw["runtime"]["config"]["num_warps"] = num_warps
    raw["id"] = sha256_json(_kernel_identity(raw))

    assert Kernel.from_dict(raw).runtime["config"]["num_warps"] == num_warps


@pytest.mark.parametrize(
    ("field", "value"),
    [("num_ctas", 2), ("num_stages", 1), ("num_stages", 32)],
)
def test_kernel_runtime_accepts_bounded_nonfixture_config_counts(field, value):
    raw = _kernel().to_dict()
    raw["runtime"]["config"][field] = value
    raw["id"] = sha256_json(_kernel_identity(raw))

    assert Kernel.from_dict(raw).runtime["config"][field] == value


@pytest.mark.parametrize(
    "edit",
    [
        lambda runtime: runtime.update(wrapper_type="not.Autotuner"),
        lambda runtime: runtime.update(leaf_type="not.JITFunction"),
        lambda runtime: runtime["config"].update(kwargs=[]),
        lambda runtime: runtime["config"].update(kwargs={"BLOCK_SIZE": 128}),
        lambda runtime: runtime["config"].update(num_warps=0),
        lambda runtime: runtime["config"].update(num_warps=3),
        lambda runtime: runtime["config"].update(num_warps=33),
        lambda runtime: runtime["config"].update(num_warps=[]),
        lambda runtime: runtime["config"].update(num_ctas=0),
        lambda runtime: runtime["config"].update(num_ctas=33),
        lambda runtime: runtime["config"].update(num_ctas=True),
        lambda runtime: runtime["config"].update(num_stages=0),
        lambda runtime: runtime["config"].update(num_stages=33),
        lambda runtime: runtime["config"].update(num_stages=True),
        lambda runtime: runtime["config"].update(num_stages=3.0),
        lambda runtime: runtime["config"].update(pre_hook="hook"),
        lambda runtime: runtime["config"].update(ir_override="override"),
        lambda runtime: runtime["config"].update(unexpected=None),
        lambda runtime: runtime.update(jit_cache_key="A" * 64),
    ],
)
def test_kernel_runtime_rejects_rehashed_noncanonical_profiles(edit):
    raw = _kernel().to_dict()
    edit(raw["runtime"])
    raw["id"] = sha256_json(_kernel_identity(raw))

    with pytest.raises(InvalidArtifact):
        Kernel.from_dict(raw)


def _kernel() -> Kernel:
    source = """def copied(X_ptr, Y_ptr):
    value = tl.load(X_ptr)
    tl.store(Y_ptr, value)
"""
    runtime = {
        "wrapper_type": "triton.runtime.autotuner.Autotuner",
        "leaf_type": "triton.runtime.jit.JITFunction",
        "config": {
            "kwargs": {},
            "num_warps": 4,
            "num_ctas": 1,
            "num_stages": 3,
            "maxnreg": None,
            "pre_hook": None,
            "ir_override": None,
        },
        "jit_cache_key": "0" * 64,
    }
    source_sha256 = sha256_bytes(source.encode("utf-8"))
    identity = {
        "module": "tests.fixture",
        "qualname": "copied",
        "source_sha256": source_sha256,
        "arguments": ["X_ptr", "Y_ptr"],
        "reads": ["X_ptr"],
        "writes": ["Y_ptr"],
        "runtime": runtime,
    }
    return Kernel(
        id=sha256_json(identity),
        module="tests.fixture",
        qualname="copied",
        source_sha256=source_sha256,
        source=source,
        argument_names=("X_ptr", "Y_ptr"),
        reads=("X_ptr",),
        writes=("Y_ptr",),
        source_reads=("X_ptr",),
        source_writes=("Y_ptr",),
        runtime=runtime,
        )

def _kernel_identity(raw):
    return {
        "module": raw["module"],
        "qualname": raw["qualname"],
        "source_sha256": raw["source_sha256"],
        "arguments": raw["argument_names"],
        "reads": raw["reads"],
        "writes": raw["writes"],
        "runtime": raw["runtime"],
    }


def test_v2_analysis_preserves_borrowed_write_subclass_serialization():
    from dataclasses import replace

    from daggertrace.model import BorrowedWrite
    from daggertrace.ownership import analyze

    class BorrowedWriteSubclass(BorrowedWrite):
        pass

    candidate, control = _micrograph(True, "direct", False, 1, False)
    result = analyze(candidate, control)
    inherited = BorrowedWriteSubclass(**result.borrowed_writes[0].to_dict())

    assert replace(result, borrowed_writes=(inherited,)).to_dict() == result.to_dict()


def test_retired_capture_schema_is_rejected_before_shape_admission():
    capture, _ = _micrograph(False, "none", False, 1, False)
    raw = capture.to_dict()
    raw["schema"] = "daggertrace.aot-capture/3"

    with pytest.raises(InvalidArtifact, match="unsupported capture schema"):
        Capture.from_dict(raw)
