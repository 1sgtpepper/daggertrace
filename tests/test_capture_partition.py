from __future__ import annotations

from types import SimpleNamespace

import pytest

from daggertrace import capture
from daggertrace.capture import _partition_hop_arguments
from daggertrace.errors import UnsupportedCapture


def test_hop_arguments_separate_reachable_inline_constexpr_values_from_tensors():
    tensor_node = object()

    bindings, constants = _partition_hop_arguments(
        {
            "X_ptr": tensor_node,
            "columns": 257,
            "MIN_LOG_SCALE": -8.0,
            "MAX_LOG_SCALE": 8.0,
            "BLOCK_SIZE": 512,
        },
        {tensor_node: "n0"},
    )

    assert bindings == {"X_ptr": tensor_node}
    assert constants == {
        "columns": 257,
        "MIN_LOG_SCALE": -8.0,
        "MAX_LOG_SCALE": 8.0,
        "BLOCK_SIZE": 512,
    }


class _FXNode:
    def __init__(self, op, descriptor):
        self.op = op
        self.args = ()
        self.kwargs = {}
        self.meta = {"desc": descriptor}


@pytest.fixture
def descriptor_graph(monkeypatch):
    placeholder = _FXNode("placeholder", None)
    output = _FXNode("output", [{"kind": "output", "index": 0}])
    monkeypatch.setattr(capture, "_target", lambda node: node.op)
    monkeypatch.setattr(capture, "_value", lambda value, _ids: value)
    monkeypatch.setattr(capture, "_descriptor", lambda value: value)
    monkeypatch.setattr(capture, "_tensor_meta", lambda _node, _aliases: None)
    return SimpleNamespace(graph=SimpleNamespace(nodes=[placeholder, output]))


@pytest.mark.parametrize("descriptor", [[], [{"kind": "primal", "index": 0}]])
def test_normalization_rejects_nested_placeholder_descriptors(descriptor_graph, descriptor):
    descriptor_graph.graph.nodes[0].meta["desc"] = descriptor

    with pytest.raises(UnsupportedCapture, match="placeholder descriptor is not a role"):
        capture._normalize_graph(descriptor_graph, "forward", {})


@pytest.mark.parametrize("descriptor", [None, {}, [None], [[]], [
    {"kind": "output", "index": 0}, None,
]])
def test_normalization_rejects_output_descriptors_without_dropping_entries(
    descriptor_graph, descriptor
):
    descriptor_graph.graph.nodes[-1].meta["desc"] = descriptor

    with pytest.raises(UnsupportedCapture, match="output descriptor is not a role list"):
        capture._normalize_graph(descriptor_graph, "forward", {})


@pytest.mark.parametrize("descriptor", [None, {"kind": "primal", "index": 0}])
def test_normalization_preserves_saved_and_described_placeholders(descriptor_graph, descriptor):
    descriptor_graph.graph.nodes[0].meta["desc"] = descriptor

    graph = capture._normalize_graph(descriptor_graph, "forward", {})

    assert graph.nodes[0].role == descriptor
    assert graph.output_roles == ({"kind": "output", "index": 0},)
