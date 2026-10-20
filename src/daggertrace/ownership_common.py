from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import InvalidArtifact, UnsupportedCapture
from .model import Capture, Node, Reason
from .model_core import _fresh_layout_stride


@dataclass(frozen=True, slots=True)
class _State:
    buffer: str
    version: int
    parts: tuple[str, ...]
    borrowed_tangent: str | None
    producer: str


@dataclass(frozen=True, slots=True)
class _GraphResult:
    gradients: dict[int, _State]
    borrowed_writes: tuple[Any, ...]
    borrowed_parts: dict[tuple[str, str], str]


_ALIASES = {"aten.view", "aten._unsafe_view", "aten.alias", "aten.detach"}
_ALLOCATIONS = {"aten.empty_like", "aten.zeros", "aten.zeros_like", "aten.empty"}
_ALLOCATION_KWARGS = {"dtype", "layout", "device", "pin_memory", "memory_format"}
_PURE = {"aten.mul", "aten.sum"}
_FRESH_RESULTS = _ALLOCATIONS | _PURE | {
    "aten.clone",
    "aten.add",
    "operator.getitem",
}
_CALL_TARGETS = _ALIASES | _ALLOCATIONS | _PURE | {
    "aten.clone",
    "aten.add",
    "operator.getitem",
    "triton.functional",
}
_MAX_DERIVED_PARTS = 4096
_ALLOWED_DTYPES = frozenset(
    {
        "bool", "uint8", "uint16", "uint32", "uint64",
        "int8", "int16", "int32", "int64",
        "float16", "bfloat16", "float32", "float64",
        "complex64", "complex128",
    }
)


def _safe_capture_digest(capture: Capture) -> str:
    try:
        return capture.digest()
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise InvalidArtifact("capture cannot be canonically digested") from exc

def _canonical_storage_names(nodes) -> dict[str, str]:
    names: dict[str, str] = {}
    for node in nodes:
        if node.tensor is None or node.tensor.alias_group is None:
            continue
        names.setdefault(node.tensor.alias_group, f"storage:a{len(names)}")
    return names


def _forward_output_domains(capture: Capture) -> tuple[set[int], set[int]]:
    slots: set[int] = set()
    saved_slots: set[int] = set()
    for role in capture.forward.output_roles:
        if not isinstance(role, dict):
            raise UnsupportedCapture(f"malformed forward output descriptor {role!r}")
        if role.get("kind") == "saved_output":
            index = _role_index(role, "saved_output", "forward saved-output descriptor")
            if index in saved_slots:
                raise UnsupportedCapture(f"duplicate forward saved-output slot {index}")
            saved_slots.add(index)
            continue
        if role.get("kind") != "output":
            raise UnsupportedCapture(f"unsupported forward output descriptor {role!r}")
        index = _role_index(role, "output", "forward output descriptor")
        if index in slots:
            raise UnsupportedCapture(f"duplicate forward output slot {index}")
        slots.add(index)
    return slots, saved_slots


def _forward_primal_slots(capture: Capture) -> set[int]:
    slots: set[int] = set()
    for node in capture.forward.nodes:
        if node.op != "placeholder":
            continue
        if node.role is None:
            raise UnsupportedCapture(f"forward placeholder {node.id} lacks a primal descriptor")
        index = _role_index(node.role, "primal", f"forward placeholder {node.id}")
        if index in slots:
            raise UnsupportedCapture(f"duplicate forward primal slot {index}")
        slots.add(index)
    return slots


def _role_index(role: Any, kind: str, label: str) -> int:
    if not isinstance(role, dict) or set(role) != {"kind", "index"} or role.get("kind") != kind:
        raise UnsupportedCapture(f"unsupported {label} {role!r}")
    index = role["index"]
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise UnsupportedCapture(f"unsupported {label} index {index!r}")
    return index


def _tangent_identity(
    role: Any,
    *,
    tangent_outputs: set[int],
    saved_outputs: set[int],
    primal_inputs: set[int],
    label: str,
) -> str | None:
    if not isinstance(role, dict):
        raise UnsupportedCapture(f"{label} has a malformed AOT descriptor")
    kind = role.get("kind")
    if kind == "unsupported":
        raise UnsupportedCapture(f"{label} has an unsupported AOT descriptor")
    if kind == "primal":
        index = _role_index(role, "primal", f"{label} primal descriptor")
        if index not in primal_inputs:
            raise UnsupportedCapture(f"{label} primal slot {index} is outside the primal domain")
        return None
    if kind == "saved_output":
        index = _role_index(role, "saved_output", f"{label} saved-output descriptor")
        if index not in saved_outputs:
            raise UnsupportedCapture(
                f"{label} saved-output slot {index} is outside the saved-output domain"
            )
        return None
    if kind != "tangent" or set(role) != {"kind", "output"}:
        raise UnsupportedCapture(f"{label} has an unsupported AOT descriptor {role!r}")
    index = _role_index(role["output"], "output", f"{label} tangent descriptor")
    if index not in tangent_outputs:
        raise UnsupportedCapture(f"{label} tangent slot {index} is outside the output domain")
    return f"tangent:output:{index}"


def _one_state_reference(node: Node, states: dict[str, _State]) -> str:
    dependencies = _all_state_references(node.args, states)
    if len(dependencies) != 1:
        raise UnsupportedCapture(f"alias node {node.id} has {len(dependencies)} tensor inputs")
    return dependencies[0]


def _one_state(node: Node, states: dict[str, _State]) -> _State:
    return states[_one_state_reference(node, states)]


def _all_state_references(value: Any, states: dict[str, _State]) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        if set(value) == {"ref"}:
            node_id = _ref_id(value, "tensor reference")
            if node_id in states:
                result.append(node_id)
            return result
        for child in value.values():
            result.extend(_all_state_references(child, states))
    elif isinstance(value, list):
        for child in value:
            result.extend(_all_state_references(child, states))
    return result


def _all_states(value: Any, states: dict[str, _State]) -> list[_State]:
    return [states[node_id] for node_id in _all_state_references(value, states)]


def _state_from_ref(value: Any, states: dict[str, _State], label: str) -> _State:
    node_id = _ref_id(value, label)
    if node_id not in states:
        raise UnsupportedCapture(f"{label} refers to non-tensor node {node_id}")
    return states[node_id]


def _ref_id(value: Any, label: str) -> str:
    if not isinstance(value, dict) or set(value) != {"ref"} or not isinstance(value["ref"], str):
        raise UnsupportedCapture(f"{label} is not a node reference")
    return value["ref"]


def _positional(value: Any) -> list[Any]:
    if isinstance(value, dict) and set(value) == {"tuple"} and isinstance(value["tuple"], list):
        return value["tuple"]
    raise UnsupportedCapture("captured positional arguments lost tuple structure")


def _validate_allocation_grammar(node: Node, nodes: tuple[Node, ...]) -> None:
    values = _positional(node.args)
    if len(values) != 1:
        raise UnsupportedCapture(
            f"allocation node {node.id} requires exactly one positional argument"
        )
    if not isinstance(node.kwargs, dict) or not all(
        isinstance(key, str) and key in _ALLOCATION_KWARGS for key in node.kwargs
    ):
        raise UnsupportedCapture(f"allocation node {node.id} has unsupported keyword arguments")
    if _contains_encoded_ref(node.kwargs):
        raise UnsupportedCapture(f"allocation node {node.id} keyword arguments contain tensor refs")
    for key, value in node.kwargs.items():
        if key == "pin_memory":
            if type(value) is not bool:
                raise UnsupportedCapture(
                    f"allocation node {node.id} keyword {key} is malformed"
                )
            continue
        if type(value) is dict and set(value) == {"torch"}:
            token = value["torch"]
        elif key == "device" and type(value) is str:
            # v2 captures may retain the legacy direct device spelling.
            token = value
        else:
            raise UnsupportedCapture(
                f"allocation node {node.id} keyword {key} is malformed"
            )
        if type(token) is not str or not token or len(token.encode("utf-8")) > 128:
            raise UnsupportedCapture(
                f"allocation node {node.id} keyword {key} is malformed"
            )
        normalized = token.removeprefix("torch.")
        if (
            key == "dtype" and normalized not in _ALLOWED_DTYPES
        ) or (
            key == "layout" and normalized != "strided"
        ) or (
            key == "memory_format"
            and normalized not in {"contiguous_format", "preserve_format"}
        ):
            raise UnsupportedCapture(
                f"allocation node {node.id} keyword {key} is unsupported"
            )
        if key == "device" and token.startswith("cuda:"):
            suffix = token[5:]
            if (
                not suffix.isascii()
                or not suffix.isdigit()
                or str(int(suffix)) != suffix
                or int(suffix) > 65535
            ):
                raise UnsupportedCapture(
                    f"allocation node {node.id} keyword {key} is malformed"
                )

    argument = values[0]
    if node.target in {"aten.empty", "aten.zeros"}:
        if not {"dtype", "device"} <= set(node.kwargs):
            raise UnsupportedCapture(
                f"{node.target} allocation {node.id} must declare dtype and device"
            )
        if type(argument) is not list or len(argument) > 64:
            raise UnsupportedCapture(f"{node.target} allocation {node.id} size is malformed")
        elements = 1
        for dimension in argument:
            if type(dimension) is not int or not 0 <= dimension <= (1 << 63) - 1:
                raise UnsupportedCapture(f"{node.target} allocation {node.id} size is malformed")
            elements *= dimension
            if elements > (1 << 63) - 1:
                raise UnsupportedCapture(
                    f"{node.target} allocation {node.id} size product exceeds signed-64"
                )
        expected_dtype = node.kwargs["dtype"]["torch"].removeprefix("torch.")
        device = node.kwargs["device"]
        expected_device = device["torch"] if isinstance(device, dict) else device
        if node.tensor is None or any(
            getattr(node.tensor, key) != expected
            for key, expected in (
                ("shape", tuple(argument)),
                ("dtype", expected_dtype),
                ("device", expected_device),
            )
        ):
            raise UnsupportedCapture(
                f"{node.target} allocation {node.id} result metadata changed"
            )
        expected_stride = _fresh_layout_stride(
            tuple(argument), memory_format=node.kwargs.get("memory_format")
        )
        if (
            node.tensor.storage_offset != 0
            or expected_stride is None
            or node.tensor.stride != expected_stride
        ):
            raise UnsupportedCapture(
                f"{node.target} allocation {node.id} result layout changed"
            )
        return

    reference = _ref_id(argument, f"{node.target} {node.id} input")
    node_indexes = {item.id: index for index, item in enumerate(nodes)}
    if (
        reference not in node_indexes
        or node_indexes[reference] >= node_indexes[node.id]
        or nodes[node_indexes[reference]].tensor is None
    ):
        raise UnsupportedCapture(f"{node.target} {node.id} requires one prior tensor reference")
    if node.target in {"aten.empty_like", "aten.zeros_like"}:
        source = nodes[node_indexes[reference]]
        expected_dtype = source.tensor.dtype
        expected_device = source.tensor.device
        for key in ("dtype", "device"):
            if key not in node.kwargs:
                continue
            override = node.kwargs[key]
            if isinstance(override, dict):
                override = override["torch"]
            if key == "dtype":
                expected_dtype = override.removeprefix("torch.")
            else:
                expected_device = override
        if node.tensor is None or any(
            getattr(node.tensor, key) != expected
            for key, expected in (
                ("shape", source.tensor.shape),
                ("dtype", expected_dtype),
                ("device", expected_device),
            )
        ):
            raise UnsupportedCapture(f"{node.target} allocation {node.id} result metadata changed")
        expected_stride = _fresh_layout_stride(
            source.tensor.shape,
            source.tensor.shape,
            source.tensor.stride,
            node.kwargs.get("memory_format"),
        )
        if (
            node.tensor.storage_offset != 0
            or expected_stride is None
            or node.tensor.stride != expected_stride
        ):
            raise UnsupportedCapture(f"{node.target} allocation {node.id} result layout changed")


def _allocation_signature(
    node: Node, states: dict[str, _State], capture_digest: str
) -> dict[str, Any]:
    if node.tensor is None:
        raise UnsupportedCapture(f"allocation node {node.id} lacks tensor metadata")
    dependencies = []
    for reference in _all_state_references(node.args, states):
        state = states[reference]
        dependencies.append(
            {
                "node": reference,
                "buffer": state.buffer,
                "version": state.version,
                "parts": list(state.parts),
            }
        )
    layout = node.tensor.to_dict()
    layout.pop("alias_group")
    signature = {
        "capture_digest": capture_digest,
        "target": node.target,
        "args": {
            "dependencies": dependencies,
            "literals": _without_refs(node.args),
        },
        "kwargs": node.kwargs,
        "layout": layout,
        "node": node.id,
    }
    return signature


def _contains_encoded_ref(value: Any) -> bool:
    if isinstance(value, dict):
        if "ref" in value:
            return True
        return any(_contains_encoded_ref(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_encoded_ref(child) for child in value)
    return False


def _graph_output_refs(value: Any) -> list[Any]:
    outer = _positional(value)
    if len(outer) != 1:
        raise UnsupportedCapture("FX output node must have one aggregate argument")
    aggregate = outer[0]
    return _positional(aggregate)


def _without_refs(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"ref"}:
            return "<tensor>"
        return {key: _without_refs(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_without_refs(child) for child in value]
    return value


def _layout_for_binding(nodes: dict[str, Node], value: Any, label: str) -> dict[str, Any]:
    node_id = _ref_id(value, label)
    tensor = nodes[node_id].tensor
    if tensor is None:
        raise UnsupportedCapture(f"{label} lacks tensor metadata")
    layout = tensor.to_dict()
    layout.pop("alias_group")
    return layout


def _unknown_reason(exc: Exception) -> Reason:
    cause: BaseException | None = exc.__cause__
    source_profile = False
    while cause is not None:
        if isinstance(cause, UnsupportedCapture):
            message = str(cause)
            source_profile = True
            break
        cause = cause.__cause__
    else:
        message = str(exc)
    for reason in (
        Reason.MEMORY_MODEL,
        Reason.FRAMEWORK_PROFILE,
        Reason.TWIN_RELATION,
        Reason.SOURCE_EFFECT,
        Reason.SOURCE_PROFILE,
        Reason.REGION_ALIAS,
    ):
        if message.startswith(f"{reason.value}:"):
            return reason
    return Reason.SOURCE_PROFILE if source_profile else Reason.GRAPH_EFFECT
