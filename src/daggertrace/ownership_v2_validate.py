from __future__ import annotations

import math
from typing import Any

from .errors import UnsupportedCapture
from .model import (
    Capture,
    Node,
    Reason,
    walk_refs,
)
from .model_core import _fresh_layout_stride, _view_result_layout
from .ownership_common import (
    _ALIASES,
    _ALLOCATIONS,
    _CALL_TARGETS,
    _FRESH_RESULTS,
    _contains_encoded_ref,
    _graph_output_refs,
    _positional,
    _ref_id,
    _validate_allocation_grammar,
)
from .version import CHECKER_PROFILE


def _validate_v2_diagnostics(diagnostics: Any, primal_inputs: set[int]) -> set[int]:
    fields = {"callbacks", "aot_autograd", "inputs", "grad_output"}
    if not isinstance(diagnostics, dict) or set(diagnostics) != fields:
        raise UnsupportedCapture("capture diagnostics fields are malformed")
    callbacks = diagnostics["callbacks"]
    if (
        not isinstance(callbacks, dict)
        or set(callbacks) != {"forward", "backward"}
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value != 1
            for value in callbacks.values()
        )
    ):
        raise UnsupportedCapture("capture diagnostics callbacks are malformed")
    counters = diagnostics["aot_autograd"]
    if (
        not isinstance(counters, dict)
        or not counters
        or any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for key, value in counters.items()
        )
    ):
        raise UnsupportedCapture("capture diagnostics aot_autograd is malformed")
    inputs = diagnostics["inputs"]
    if (
        not isinstance(inputs, list)
        or not inputs
        or len(inputs) > 64
        or set(range(len(inputs))) != primal_inputs
    ):
        raise UnsupportedCapture("capture diagnostics inputs are malformed")
    expected_gradients: set[int] = set()
    for index, item in enumerate(inputs):
        _validate_v2_runtime_tensor(item, f"capture diagnostics inputs[{index}]")
        if item["requires_grad"]:
            expected_gradients.add(index)
    if not expected_gradients:
        raise UnsupportedCapture("capture diagnostics contain no differentiable inputs")
    _validate_v2_runtime_tensor(diagnostics["grad_output"], "capture diagnostics grad_output")
    return expected_gradients


def _validate_v2_runtime_tensor(raw: Any, label: str) -> None:
    fields = {"shape", "stride", "storage_offset", "dtype", "device", "requires_grad"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise UnsupportedCapture(f"{label} is malformed")
    shape = raw["shape"]
    stride = raw["stride"]
    if (
        not isinstance(shape, list)
        or not isinstance(stride, list)
        or len(shape) != len(stride)
        or len(shape) > 64
        or any(
            not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < (1 << 63)
            for value in shape
        )
        or any(
            not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < (1 << 63)
            for value in stride
        )
    ):
        raise UnsupportedCapture(f"{label} layout is malformed")
    elements = 1
    for dimension in shape:
        elements *= dimension
        if elements > (1 << 63) - 1:
            raise UnsupportedCapture(f"{label} shape product exceeds signed-64")
    if (
        not isinstance(raw["storage_offset"], int)
        or isinstance(raw["storage_offset"], bool)
        or not 0 <= raw["storage_offset"] < (1 << 63)
    ):
        raise UnsupportedCapture(f"{label} storage offset is malformed")
    if not isinstance(raw["dtype"], str) or not raw["dtype"]:
        raise UnsupportedCapture(f"{label} dtype is malformed")
    if not isinstance(raw["device"], str) or not raw["device"]:
        raise UnsupportedCapture(f"{label} device is malformed")
    if not isinstance(raw["requires_grad"], bool):
        raise UnsupportedCapture("capture diagnostics requires_grad must be boolean")


def _validate_capture_profile(capture: Capture) -> None:
    Capture.from_dict(capture)
    framework_keys = {"python", "torch", "torch_git", "triton", "cuda_runtime"}
    expected = {
        "torch": "2.12.1+cu130",
        "torch_git": "7269437d655783a26cba32aa88195b741ff496aa",
        "triton": "3.7.1",
    }
    if not isinstance(capture.framework, dict) or set(capture.framework) != framework_keys:
        raise UnsupportedCapture(f"{Reason.FRAMEWORK_PROFILE.value}: framework fields changed")
    for key, value in expected.items():
        if capture.framework.get(key) != value:
            raise UnsupportedCapture(f"{Reason.FRAMEWORK_PROFILE.value}: {key}={capture.framework.get(key)!r}")
    for key in ("python", "cuda_runtime"):
        value = capture.framework[key]
        if not isinstance(value, str) or not value:
            raise UnsupportedCapture(f"{Reason.FRAMEWORK_PROFILE.value}: {key} is malformed")
    if capture.capture_config != {
        "fullgraph": True,
        "dynamic": False,
        "stage": "post-dynamo-pre-inductor",
    }:
        raise UnsupportedCapture(f"capture configuration is outside {CHECKER_PROFILE}")
    if capture.forward.kind != "forward" or capture.backward.kind != "backward":
        raise UnsupportedCapture("capture graph kinds are invalid")
    kernels = {kernel.id: kernel for kernel in capture.kernels}
    unsupported = [
        node.target
        for graph in (capture.forward, capture.backward)
        for node in graph.nodes
        if node.target.startswith("unsupported:")
    ]
    if unsupported:
        raise UnsupportedCapture(f"unsupported captured targets: {sorted(set(unsupported))}")
    for graph in (capture.forward, capture.backward):
        _validate_graph_grammar(graph)
        _validate_storage_identities(graph)
        for node in graph.nodes:
            if node.gradient_accumulation and node.target != "aten.add":
                raise UnsupportedCapture(
                    f"gradient-accumulation metadata appears on non-add node {node.id}"
                )
    _validate_forward_hops(capture)
    for node in capture.backward.nodes:
        if node.target == "triton.functional":
            kernel = kernels.get(node.kwargs.get("kernel"))
            if kernel is None or node.kwargs.get("writes") != sorted(kernel.source_writes):
                raise UnsupportedCapture(f"backward HOP {node.id} writes differ from source effects")
    referenced = {
        node.kwargs.get("kernel")
        for graph in (capture.forward, capture.backward)
        for node in graph.nodes
        if node.target == "triton.functional" and isinstance(node.kwargs, dict)
    }
    if referenced != {kernel.id for kernel in capture.kernels}:
        raise UnsupportedCapture("captured kernel table is not referentially total")


def _validate_forward_hops(capture: Capture) -> None:
    kernels = {kernel.id: kernel for kernel in capture.kernels}
    nodes = {node.id: node for node in capture.forward.nodes}
    hop_writes: dict[str, set[str]] = {}
    for node in capture.forward.nodes:
        if node.target == "triton.functional":
            kernel = kernels.get(node.kwargs["kernel"])
            if kernel is None:
                raise UnsupportedCapture(f"forward HOP {node.id} references an unknown kernel")
            bindings = node.kwargs["bindings"]
            constants = node.kwargs["constants"]
            writes = node.kwargs["writes"]
            required_roles = set(kernel.reads) | set(kernel.writes)
            if (
                not isinstance(bindings, dict)
                or set(bindings) != required_roles
                or not isinstance(constants, dict)
                or set(bindings) & set(constants)
                or set(bindings) | set(constants) != set(kernel.argument_names)
                or _contains_encoded_ref(constants)
            ):
                raise UnsupportedCapture(
                    f"forward HOP {node.id} is not total over its source-bound signature"
                )
            if writes != sorted(kernel.source_writes):
                raise UnsupportedCapture(
                    f"forward HOP {node.id} writes differ from source effects"
                )
            write_references = []
            write_groups = []
            for role, value in bindings.items():
                reference = _ref_id(value, f"forward HOP {node.id}.{role}")
                bound = nodes.get(reference)
                if bound is None or bound.tensor is None:
                    raise UnsupportedCapture(
                        f"forward HOP {node.id}.{role} is not bound to a tensor"
                    )
                if role in kernel.source_writes:
                    write_references.append(reference)
                    write_groups.append(bound.tensor.alias_group)
            if len(write_references) != len(set(write_references)) or len(write_groups) != len(
                set(write_groups)
            ):
                raise UnsupportedCapture(f"forward HOP {node.id} has aliased write destinations")
            hop_writes[node.id] = set(writes)
        elif node.target == "operator.getitem":
            values = _positional(node.args)
            if len(values) != 2 or node.kwargs != {}:
                raise UnsupportedCapture(f"forward getitem {node.id} grammar changed")
            hop = _ref_id(values[0], f"forward getitem {node.id}")
            key = values[1]
            if not isinstance(key, str) or key not in hop_writes.get(hop, set()):
                raise UnsupportedCapture(
                    f"forward getitem {node.id} does not select a declared HOP write"
                )


def _validate_storage_identities(graph) -> None:
    nodes = {node.id: node for node in graph.nodes}
    storage_roots: dict[str, str] = {}
    for node in graph.nodes:
        if node.op != "placeholder" and node.target not in _FRESH_RESULTS:
            continue
        group = node.tensor.alias_group if node.tensor is not None else None
        if group is None:
            label = "placeholder" if node.op == "placeholder" else "fresh result"
            raise UnsupportedCapture(
                f"{Reason.REGION_ALIAS.value}: {label} {node.id} lacks captured storage identity"
            )
        prior = storage_roots.get(group)
        if prior is not None:
            prior_node = nodes[prior]
            if node.op == "placeholder" and prior_node.op == "placeholder":
                raise UnsupportedCapture(
                    f"{Reason.REGION_ALIAS.value}: distinct placeholders {prior}/{node.id} share storage"
                )
            if node.op == "placeholder":
                raise UnsupportedCapture(
                    f"{Reason.REGION_ALIAS.value}: placeholder {node.id} reuses storage from {prior}"
                )
            raise UnsupportedCapture(
                f"{Reason.REGION_ALIAS.value}: fresh result {node.id} reuses storage from {prior}"
            )
        storage_roots[group] = node.id

    for node in graph.nodes:
        if node.target not in _ALIASES:
            continue
        references = [
            reference
            for reference in walk_refs(node.args)
            if reference in nodes and nodes[reference].tensor is not None
        ]
        if len(references) != 1:
            raise UnsupportedCapture(
                f"{Reason.REGION_ALIAS.value}: alias node {node.id} has "
                f"{len(references)} tensor inputs"
            )
        source_id = references[0]
        source_tensor = nodes[source_id].tensor
        if (
            node.tensor is None
            or source_tensor is None
            or node.tensor.alias_group != source_tensor.alias_group
            or (
                node.target in {"aten.alias", "aten.detach"}
                and any(
                    getattr(node.tensor, key) != getattr(source_tensor, key)
                    for key in ("shape", "stride", "dtype", "device", "storage_offset")
                )
            )
        ):
            raise UnsupportedCapture(
                f"{Reason.REGION_ALIAS.value}: alias node {node.id} does not preserve "
                f"storage identity from {source_id}"
            )


def _validate_fresh_result_metadata(
    node: Node, shape: tuple[int, ...], dtype: str, device: str
) -> None:
    expected_stride = _fresh_layout_stride(shape)
    if node.tensor is None or any(
        getattr(node.tensor, key) != expected
        for key, expected in (("shape", shape), ("dtype", dtype), ("device", device))
    ):
        raise UnsupportedCapture(f"{node.target} node {node.id} result metadata changed")
    if (
        expected_stride is None
        or node.tensor.storage_offset != 0
        or node.tensor.stride != expected_stride
    ):
        raise UnsupportedCapture(f"{node.target} node {node.id} result layout changed")


def _validate_view_result_metadata(node: Node, source: Node, requested: list[int]) -> None:
    if node.tensor is None or source.tensor is None or any(
        getattr(node.tensor, key) != getattr(source.tensor, key)
        for key in ("alias_group", "dtype", "device", "storage_offset")
    ):
        raise UnsupportedCapture(f"view node {node.id} result storage identity changed")
    layout = _view_result_layout(source.tensor.shape, source.tensor.stride, requested)
    if layout is None or (node.tensor.shape, node.tensor.stride) != layout:
        raise UnsupportedCapture(f"view node {node.id} result layout changed")


def _validate_graph_grammar(graph) -> None:
    nodes_by_id = {item.id: item for item in graph.nodes}
    node_indexes = {item.id: index for index, item in enumerate(graph.nodes)}
    selections: set[tuple[str, str]] = set()
    for node in graph.nodes:
        index = node_indexes[node.id]
        if node.tensor is not None:
            if len(node.tensor.shape) > 64 or len(node.tensor.shape) != len(node.tensor.stride):
                raise UnsupportedCapture(f"node {node.id} tensor rank is outside the bound")
            elements = 1
            for dimension in node.tensor.shape:
                if not 0 <= dimension <= (1 << 63) - 1:
                    raise UnsupportedCapture(f"node {node.id} tensor shape is outside signed-64")
                elements *= dimension
                if elements > (1 << 63) - 1:
                    raise UnsupportedCapture(
                        f"node {node.id} tensor shape product exceeds signed-64"
                    )
            if any(
                not 0 <= stride <= (1 << 63) - 1
                for stride in node.tensor.stride
            ) or not 0 <= node.tensor.storage_offset <= (1 << 63) - 1:
                raise UnsupportedCapture(
                    f"node {node.id} tensor metadata is outside signed-64"
                )
        if node.op == "placeholder":
            if node.target != "placeholder" or node.args != {"tuple": []} or node.kwargs != {}:
                raise UnsupportedCapture(f"placeholder {node.id} grammar changed")
        elif node.op == "output":
            if node.target != "output" or node.kwargs != {}:
                raise UnsupportedCapture(f"output {node.id} grammar changed")
            outputs = _graph_output_refs(node.args)
            if len(outputs) != len(graph.output_roles):
                raise UnsupportedCapture(f"output {node.id} values and roles differ")
            for value in outputs:
                _ref_id(value, f"output {node.id}")
        elif node.op != "call_function" or node.target not in _CALL_TARGETS:
            raise UnsupportedCapture(
                f"unsupported graph node {node.id}: {node.op}/{node.target}"
            )
        elif node.target == "triton.functional":
            if node.args != [] or not isinstance(node.kwargs, dict) or set(node.kwargs) != {
                "kernel",
                "bindings",
                "writes",
                "constants",
                "grid",
            }:
                raise UnsupportedCapture(f"HOP {node.id} grammar changed")
            grid = node.kwargs["grid"]
            if type(grid) is not list or len(grid) != 1:
                raise UnsupportedCapture(f"HOP {node.id} grid grammar changed")
            dimensions = grid[0]
            if (
                type(dimensions) is not dict
                or set(dimensions) != {"tuple"}
                or type(dimensions["tuple"]) is not list
                or len(dimensions["tuple"]) != 3
            ):
                raise UnsupportedCapture(f"HOP {node.id} grid grammar changed")
            product = 1
            for dimension in dimensions["tuple"]:
                if type(dimension) is not int or not 1 <= dimension <= (1 << 63) - 1:
                    raise UnsupportedCapture(f"HOP {node.id} grid grammar changed")
                product *= dimension
                if product > (1 << 63) - 1:
                    raise UnsupportedCapture(
                        f"HOP {node.id} grid product exceeds signed-64"
                    )
        else:
            values = _positional(node.args)
            if node.target in _ALLOCATIONS:
                _validate_allocation_grammar(node, graph.nodes)
            elif node.target in {"aten.view", "aten._unsafe_view"}:
                if node.kwargs != {}:
                    raise UnsupportedCapture(
                        f"alias node {node.id} has keyword arguments"
                    )
                if len(values) != 2 or type(values[1]) is not list or not 0 < len(values[1]) <= 64:
                    raise UnsupportedCapture(f"view node {node.id} grammar changed")
                if sum(item == -1 for item in values[1]) > 1 or any(
                    type(item) is not int or not -1 <= item <= (1 << 63) - 1
                    for item in values[1]
                ):
                    raise UnsupportedCapture(f"view node {node.id} grammar changed")
                reference = _ref_id(values[0], f"view node {node.id} input")
                source = nodes_by_id.get(reference)
                if source is None or node_indexes[reference] >= index or source.tensor is None:
                    raise UnsupportedCapture(f"view node {node.id} requires one prior tensor reference")
                _validate_view_result_metadata(node, source, values[1])
            elif node.target in {"aten.alias", "aten.detach", "aten.clone"}:
                if node.kwargs != {} or len(values) != 1:
                    kind = "alias" if node.target in _ALIASES else "clone"
                    raise UnsupportedCapture(f"{kind} node {node.id} grammar changed")
                reference = _ref_id(values[0], f"{node.target} node {node.id} input")
                source = nodes_by_id.get(reference)
                if source is None or node_indexes[reference] >= index or source.tensor is None:
                    raise UnsupportedCapture(
                        f"{node.target} node {node.id} requires one prior tensor reference"
                    )
                if node.target == "aten.clone" and (
                    node.tensor is None
                    or any(
                        getattr(node.tensor, key) != getattr(source.tensor, key)
                        for key in ("shape", "dtype", "device")
                    )
                ):
                    raise UnsupportedCapture(f"clone node {node.id} result metadata changed")
                expected_stride = _fresh_layout_stride(
                    source.tensor.shape, source.tensor.shape, source.tensor.stride
                )
                if node.target == "aten.clone" and (
                    node.tensor.storage_offset != 0
                    or expected_stride is None
                    or node.tensor.stride != expected_stride
                ):
                    raise UnsupportedCapture(f"clone node {node.id} result layout changed")
            elif node.target == "aten.mul":
                if node.kwargs != {} or len(values) != 2:
                    raise UnsupportedCapture(f"mul node {node.id} grammar changed")
                sources = []
                for value in values:
                    if type(value) is dict and set(value) == {"ref"}:
                        reference = _ref_id(value, f"mul node {node.id} operand")
                        source = nodes_by_id.get(reference)
                        if source is None or node_indexes[reference] >= index or source.tensor is None:
                            raise UnsupportedCapture(
                                f"mul node {node.id} requires a prior tensor reference"
                            )
                        sources.append(source)
                    elif type(value) is int and -(1 << 63) <= value <= (1 << 63) - 1:
                        continue
                    elif type(value) is float and math.isfinite(value):
                        continue
                    else:
                        raise UnsupportedCapture(f"mul node {node.id} grammar changed")
                if not sources:
                    raise UnsupportedCapture(f"mul node {node.id} grammar changed")
                source = sources[0].tensor
                if any(
                    other.tensor is None
                    or any(
                        getattr(other.tensor, key) != getattr(source, key)
                        for key in ("shape", "dtype", "device")
                    )
                    for other in sources[1:]
                ):
                    raise UnsupportedCapture(f"mul node {node.id} operands are incompatible")
                _validate_fresh_result_metadata(
                    node, source.shape, source.dtype, source.device
                )
            elif node.target == "aten.sum":
                if node.kwargs != {} or len(values) != 2 or type(values[1]) is not list:
                    raise UnsupportedCapture(f"sum node {node.id} grammar changed")
                reference = _ref_id(values[0], f"sum node {node.id} input")
                source = nodes_by_id.get(reference)
                if source is None or node_indexes[reference] >= index or source.tensor is None:
                    raise UnsupportedCapture(f"sum node {node.id} requires one prior tensor reference")
                rank = len(source.tensor.shape)
                dimensions = values[1]
                if len(dimensions) > 64 or (not rank and dimensions) or any(
                    type(item) is not int or not -rank <= item < rank for item in dimensions
                ) or (rank and len({item % rank for item in dimensions}) != len(dimensions)):
                    raise UnsupportedCapture(f"sum node {node.id} grammar changed")
                reduced = {item % rank for item in dimensions} if rank else set()
                shape = tuple(
                    dimension
                    for axis, dimension in enumerate(source.tensor.shape)
                    if axis not in reduced
                )
                _validate_fresh_result_metadata(
                    node, shape, source.tensor.dtype, source.tensor.device
                )
            elif node.target == "aten.add":
                if node.kwargs != {} or len(values) != 2:
                    raise UnsupportedCapture(f"add node {node.id} grammar changed")
                operands = []
                for value in values:
                    reference = _ref_id(value, f"add node {node.id} operand")
                    source = nodes_by_id.get(reference)
                    if source is None or node_indexes[reference] >= index or source.tensor is None:
                        raise UnsupportedCapture(
                            f"add node {node.id} requires two prior tensor references"
                        )
                    operands.append(source.tensor)
                if any(
                    getattr(operands[0], key) != getattr(operands[1], key)
                    for key in ("shape", "dtype", "device")
                ):
                    raise UnsupportedCapture(f"add node {node.id} operands are incompatible")
                _validate_fresh_result_metadata(
                    node, operands[0].shape, operands[0].dtype, operands[0].device
                )
            elif node.target == "operator.getitem":
                if node.kwargs != {} or len(values) != 2 or type(values[1]) is not str:
                    raise UnsupportedCapture(f"getitem node {node.id} grammar changed")
                hop_ref = _ref_id(values[0], f"getitem node {node.id} source")
                hop = nodes_by_id.get(hop_ref)
                if (
                    hop is None
                    or node_indexes[hop_ref] >= index
                    or hop.target != "triton.functional"
                    or not isinstance(hop.kwargs, dict)
                    or values[1] not in hop.kwargs.get("writes", [])
                    or not isinstance(hop.kwargs.get("bindings"), dict)
                ):
                    raise UnsupportedCapture(f"getitem node {node.id} does not select a HOP write")
                selection = (hop_ref, values[1])
                if selection in selections:
                    raise UnsupportedCapture(f"getitem node {node.id} duplicates a HOP output selection")
                selections.add(selection)
                binding = hop.kwargs["bindings"].get(values[1])
                bound_ref = _ref_id(binding, f"getitem node {node.id} binding")
                bound = nodes_by_id.get(bound_ref)
                if bound is None or bound.tensor is None or node.tensor is None or any(
                    getattr(node.tensor, field) != getattr(bound.tensor, field)
                    for field in ("shape", "stride", "dtype", "device", "storage_offset")
                ):
                    raise UnsupportedCapture(f"getitem node {node.id} result metadata changed")

        if node.op != "placeholder" and node.role is not None:
            raise UnsupportedCapture(f"node {node.id} has a role outside a placeholder")
        expects_tensor = node.op in {"placeholder", "call_function"} and node.target != "triton.functional"
        if expects_tensor != (node.tensor is not None):
            raise UnsupportedCapture(f"node {node.id} tensor-result metadata changed")
