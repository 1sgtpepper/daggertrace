from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .canonical import sha256_bytes, sha256_json
from .effects import profile_triton_source
from .errors import InvalidArtifact, UnsupportedCapture
from .version import CAPTURE_SCHEMA


# Triton 3.7.1 stores launch counts as integer Config fields.  The source
# profile admits only a finite, target-neutral language for those fields: a
# positive power-of-two warp count and positive bounded launch counts.  This is
# a syntactic bounded language, not a launch-legality proof; the target adapter
# still owns device-specific legality.  The bounds keep arbitrary Python
# values and unbounded resource claims out of the artifact.
_CONFIG_COUNT_MAX = 32


def _is_closed_config(config: dict[str, Any]) -> bool:
    def bounded_count(value: Any) -> bool:
        return type(value) is int and 1 <= value <= _CONFIG_COUNT_MAX

    num_warps = config["num_warps"]
    return (
        type(config["kwargs"]) is dict
        and config["kwargs"] == {}
        and bounded_count(num_warps)
        and (num_warps & (num_warps - 1)) == 0
        and bounded_count(config["num_ctas"])
        and bounded_count(config["num_stages"])
        and config["maxnreg"] is None
        and config["pre_hook"] is None
        and config["ir_override"] is None
    )


class Outcome(str, Enum):
    VIOLATION = "VIOLATION"
    SAFE_IN_FRAGMENT = "SAFE_IN_FRAGMENT"
    UNKNOWN = "UNKNOWN"


class Reason(str, Enum):
    BORROWED_COTANGENT_WRITE = "BORROWED_COTANGENT_WRITE"
    CONTRIBUTION_COLLAPSE = "CONTRIBUTION_COLLAPSE"
    FRESH_DESTINATION = "FRESH_DESTINATION"
    TWIN_RELATION = "TWIN_RELATION"
    GRAPH_EFFECT = "GRAPH_EFFECT"
    SOURCE_EFFECT = "SOURCE_EFFECT"
    SOURCE_PROFILE = "SOURCE_PROFILE"
    MEMORY_MODEL = "MEMORY_MODEL"
    FRAMEWORK_PROFILE = "FRAMEWORK_PROFILE"
    REGION_ALIAS = "REGION_ALIAS"


@dataclass(frozen=True, slots=True)
class TensorMeta:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    storage_offset: int = 0
    alias_group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "device": self.device,
            "storage_offset": self.storage_offset,
            "alias_group": self.alias_group,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> TensorMeta:
        obj = _closed_dict(
            raw,
            {"shape", "stride", "dtype", "device", "storage_offset", "alias_group"},
            "tensor",
        )
        shape = _int_tuple(obj["shape"], "tensor.shape", minimum=0)
        stride = _int_tuple(obj["stride"], "tensor.stride", minimum=0)
        if len(shape) != len(stride):
            raise InvalidArtifact("tensor shape and stride ranks differ")
        elements = 1
        for dimension in shape:
            elements *= dimension
            if elements > (1 << 63) - 1:
                raise InvalidArtifact("tensor shape product exceeds signed-64")
        dtype = _string(obj["dtype"], "tensor.dtype")
        device = _string(obj["device"], "tensor.device")
        if not dtype or not device:
            raise InvalidArtifact("tensor dtype and device must be nonempty strings")
        alias_group = obj["alias_group"]
        if alias_group is not None and (
            not isinstance(alias_group, str) or not alias_group
        ):
            raise InvalidArtifact("tensor.alias_group must be a string or null")
        return cls(
            shape=shape,
            stride=stride,
            dtype=dtype,
            device=device,
            storage_offset=_integer(obj["storage_offset"], "tensor.storage_offset", minimum=0),
            alias_group=alias_group,
        )


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    op: str
    target: str
    args: Any
    kwargs: Any
    role: dict[str, Any] | None = None
    tensor: TensorMeta | None = None
    gradient_accumulation: bool = False

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.id,
            "op": self.op,
            "target": self.target,
            "args": self.args,
            "kwargs": self.kwargs,
            "role": self.role,
            "tensor": self.tensor.to_dict() if self.tensor else None,
            "gradient_accumulation": self.gradient_accumulation,
        }
        return value

    @classmethod
    def from_dict(cls, raw: Any) -> Node:
        obj = _closed_dict(
            raw,
            {
                "id",
                "op",
                "target",
                "args",
                "kwargs",
                "role",
                "tensor",
                "gradient_accumulation",
            },
            "node",
        )
        node_id = _string(obj["id"], "node.id")
        if not node_id.startswith("n") or not node_id[1:].isdigit():
            raise InvalidArtifact(f"invalid node id {node_id!r}")
        role = obj["role"]
        if role is not None and not isinstance(role, dict):
            raise InvalidArtifact("node.role must be an object or null")
        gradient_accumulation = obj["gradient_accumulation"]
        if not isinstance(gradient_accumulation, bool):
            raise InvalidArtifact("node.gradient_accumulation must be boolean")
        tensor = None if obj["tensor"] is None else TensorMeta.from_dict(obj["tensor"])
        return cls(
            id=node_id,
            op=_string(obj["op"], "node.op"),
            target=_string(obj["target"], "node.target"),
            args=obj["args"],
            kwargs=obj["kwargs"],
            role=role,
            tensor=tensor,
            gradient_accumulation=gradient_accumulation,
        )


@dataclass(frozen=True, slots=True)
class Graph:
    kind: str
    nodes: tuple[Node, ...]
    output_roles: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "nodes": [node.to_dict() for node in self.nodes],
            "output_roles": list(self.output_roles),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Graph:
        obj = _closed_dict(raw, {"kind", "nodes", "output_roles"}, "graph")
        if not isinstance(obj["nodes"], list):
            raise InvalidArtifact("graph.nodes must be a list")
        nodes = tuple(Node.from_dict(item) for item in obj["nodes"])
        expected_ids = tuple(f"n{i}" for i in range(len(nodes)))
        if tuple(node.id for node in nodes) != expected_ids:
            raise InvalidArtifact("graph nodes must use dense topological n0..nN ids")
        kind = _string(obj["kind"], "graph.kind")
        ids = set(expected_ids)
        for index, node in enumerate(nodes):
            for ref in walk_refs(node.args):
                if ref not in ids or int(ref[1:]) >= index:
                    raise InvalidArtifact(f"node {node.id} has non-prior reference {ref}")
            for ref in walk_refs(node.kwargs):
                if ref not in ids or int(ref[1:]) >= index:
                    raise InvalidArtifact(f"node {node.id} has non-prior reference {ref}")
        selections: set[tuple[str, str]] = set()
        for node in nodes:
            if node.target != "operator.getitem" or not isinstance(node.args, dict):
                continue
            values = node.args.get("tuple")
            if (
                not isinstance(values, list)
                or len(values) != 2
                or not isinstance(values[0], dict)
                or set(values[0]) != {"ref"}
                or not isinstance(values[0]["ref"], str)
                or not isinstance(values[1], str)
            ):
                continue
            selection = (values[0]["ref"], values[1])
            if selection in selections:
                raise InvalidArtifact("graph duplicates a HOP output selection")
            selections.add(selection)
        roles = obj["output_roles"]
        if not isinstance(roles, list) or not all(isinstance(item, dict) for item in roles):
            raise InvalidArtifact("graph.output_roles must be a list of objects")
        if kind == "backward" and not roles:
            raise InvalidArtifact("backward output roles must be nonempty")
        saved = [role.get("index") for role in roles if role.get("kind") == "saved_output"]
        if any(type(index) is not int or index < 0 for index in saved) or len(saved) != len(set(saved)):
            raise InvalidArtifact("saved output roles contain a duplicate or malformed path")
        for node in nodes:
            if node.op == "call_function" and node.target == "triton.functional":
                _validate_v2_grid(node.kwargs.get("grid") if isinstance(node.kwargs, dict) else None, node.id)
        graph = cls(kind=kind, nodes=nodes, output_roles=tuple(roles))
        graph.output_node()
        return graph

    def output_node(self) -> Node:
        outputs = [node for node in self.nodes if node.op == "output"]
        if len(outputs) != 1 or outputs[0] is not self.nodes[-1]:
            raise InvalidArtifact("graph must end with exactly one output node")
        return outputs[0]

    def digest(self) -> str:
        return sha256_json(self.to_dict())


def _validate_v2_grid(raw: Any, label: str) -> None:
    if not isinstance(raw, list) or len(raw) != 1:
        raise InvalidArtifact(f"{label} grid must contain one launch tuple")
    encoded = raw[0]
    if not isinstance(encoded, dict) or set(encoded) != {"tuple"}:
        raise InvalidArtifact(f"{label} grid must contain an encoded launch tuple")
    dimensions = encoded["tuple"]
    if not isinstance(dimensions, list) or len(dimensions) != 3 or any(
        type(item) is not int or item < 1 for item in dimensions
    ):
        raise InvalidArtifact(f"{label} grid dimensions must be positive integers")


@dataclass(frozen=True, slots=True)
class Kernel:
    id: str
    module: str
    qualname: str
    source_sha256: str
    source: str
    argument_names: tuple[str, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    source_reads: tuple[str, ...]
    source_writes: tuple[str, ...]
    runtime: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "module": self.module,
            "qualname": self.qualname,
            "source_sha256": self.source_sha256,
            "source": self.source,
            "argument_names": list(self.argument_names),
            "reads": list(self.reads),
            "writes": list(self.writes),
            "source_reads": list(self.source_reads),
            "source_writes": list(self.source_writes),
            "runtime": self.runtime,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Kernel:
        keys = {
            "id",
            "module",
            "qualname",
            "source_sha256",
            "source",
            "argument_names",
            "reads",
            "writes",
            "source_reads",
            "source_writes",
            "runtime",
        }
        obj = _closed_dict(raw, keys, "kernel")
        kernel_id = _digest(obj["id"], "kernel.id")
        module = _string(obj["module"], "kernel.module")
        qualname = _string(obj["qualname"], "kernel.qualname")
        source_sha256 = _digest(obj["source_sha256"], "kernel.source_sha256")
        source = _string(obj["source"], "kernel.source")
        argument_names = _string_tuple(obj["argument_names"], "kernel.argument_names")
        reads = _string_tuple(obj["reads"], "kernel.reads")
        writes = _string_tuple(obj["writes"], "kernel.writes")
        if not writes:
            raise InvalidArtifact("kernel must declare a source-bound write")
        source_reads = _string_tuple(obj["source_reads"], "kernel.source_reads")
        source_writes = _string_tuple(obj["source_writes"], "kernel.source_writes")
        runtime = _json_object(obj["runtime"], "kernel.runtime")
        runtime = _closed_dict(
            runtime,
            {"wrapper_type", "leaf_type", "config", "jit_cache_key"},
            "kernel.runtime",
        )
        config = _closed_dict(
            runtime["config"],
            {
                "kwargs",
                "num_warps",
                "num_ctas",
                "num_stages",
                "maxnreg",
                "pre_hook",
                "ir_override",
            },
            "kernel.runtime.config",
        )
        if (
            runtime["wrapper_type"] != "triton.runtime.autotuner.Autotuner"
            or runtime["leaf_type"] != "triton.runtime.jit.JITFunction"
            or not _is_closed_config(config)
        ):
            raise InvalidArtifact("kernel.runtime is outside the exact admitted profile")
        _digest(runtime["jit_cache_key"], "kernel.runtime.jit_cache_key")
        if sha256_bytes(source.encode("utf-8")) != source_sha256:
            raise InvalidArtifact("kernel source SHA-256 does not match its bytes")
        try:
            profile = profile_triton_source(source, set(reads) | set(writes))
        except UnsupportedCapture as exc:
            raise InvalidArtifact(f"kernel source profile is invalid: {exc}") from exc
        if (
            profile.arguments != argument_names
            or profile.reads != reads
            or profile.writes != writes
            or source_reads != reads
            or source_writes != writes
        ):
            raise InvalidArtifact("kernel source signature/effects do not match recorded semantics")
        identity = {
            "module": module,
            "qualname": qualname,
            "source_sha256": source_sha256,
            "arguments": list(argument_names),
            "reads": list(reads),
            "writes": list(writes),
            "runtime": runtime,
        }
        if sha256_json(identity) != kernel_id:
            raise InvalidArtifact("kernel identity digest does not match recorded semantics")
        return cls(
            id=kernel_id,
            module=module,
            qualname=qualname,
            source_sha256=source_sha256,
            source=source,
            argument_names=argument_names,
            reads=reads,
            writes=writes,
            source_reads=source_reads,
            source_writes=source_writes,
            runtime=runtime,
        )


@dataclass(frozen=True, slots=True)
class Capture:
    label: str
    framework: dict[str, Any]
    capture_config: dict[str, Any]
    forward: Graph
    backward: Graph
    kernels: tuple[Kernel, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    schema: str = CAPTURE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "label": self.label,
            "framework": self.framework,
            "capture_config": self.capture_config,
            "forward": self.forward.to_dict(),
            "backward": self.backward.to_dict(),
            "kernels": [kernel.to_dict() for kernel in self.kernels],
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Capture:
        if type(raw) is Capture:
            schema = object.__getattribute__(raw, "schema")
            if _schema_is(schema, CAPTURE_SCHEMA):
                cls.from_dict(raw.to_dict())
                return raw
            if type(schema) is str:
                raise InvalidArtifact(f"unsupported capture schema {schema!r}")
            raise InvalidArtifact("unsupported capture schema")
        elif isinstance(raw, dict):
            if dict.__len__(raw) > 50_000:
                raise InvalidArtifact("capture exceeds its structural bound")
            schema = _probe_schema(raw)
            if schema is _MISSING:
                raise InvalidArtifact("capture schema must be an exact string")
            if not _schema_is(schema, CAPTURE_SCHEMA):
                if type(schema) is str:
                    raise InvalidArtifact(f"unsupported capture schema {schema!r}")
                raise InvalidArtifact("unsupported capture schema")
        try:
            obj = _closed_dict(
                raw,
                {
                    "schema",
                    "label",
                    "framework",
                    "capture_config",
                    "forward",
                    "backward",
                    "kernels",
                    "diagnostics",
                },
                "capture",
            )
            if not _schema_is(obj["schema"], CAPTURE_SCHEMA):
                if type(obj["schema"]) is str:
                    raise InvalidArtifact(f"unsupported capture schema {obj['schema']!r}")
                raise InvalidArtifact("unsupported capture schema")
            if not isinstance(obj["framework"], dict) or not isinstance(obj["capture_config"], dict):
                raise InvalidArtifact("capture framework and capture_config must be objects")
            if not isinstance(obj["diagnostics"], dict):
                raise InvalidArtifact("capture diagnostics must be an object")
            if not isinstance(obj["kernels"], list):
                raise InvalidArtifact("capture kernels must be a list")
            kernels = tuple(Kernel.from_dict(item) for item in obj["kernels"])
            if len({kernel.id for kernel in kernels}) != len(kernels):
                raise InvalidArtifact("capture kernel ids are not unique")
            capture = cls(
                schema=CAPTURE_SCHEMA,
                label=_string(obj["label"], "capture.label", nonempty=True),
                framework=obj["framework"],
                capture_config=obj["capture_config"],
                forward=Graph.from_dict(obj["forward"]),
                backward=Graph.from_dict(obj["backward"]),
                kernels=kernels,
                diagnostics=obj["diagnostics"],
            )
            _json_value(capture.to_dict(), "capture")
            return capture
        except RecursionError as exc:
            raise InvalidArtifact("capture structure is too deep or cyclic") from exc

    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class BorrowedWrite:
    tangent: str
    hop_node: str
    read_role: str
    write_role: str
    kernel_id: str
    before_version: int
    after_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tangent": self.tangent,
            "hop_node": self.hop_node,
            "read_role": self.read_role,
            "write_role": self.write_role,
            "kernel_id": self.kernel_id,
            "before_version": self.before_version,
            "after_version": self.after_version,
        }


@dataclass(frozen=True, slots=True)
class Collapse:
    gradient_of: int
    accumulation_node: str
    expected_parts: tuple[str, ...]
    actual_parts: tuple[str, ...]
    replaced_part: str
    replacement_part: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gradient_of": self.gradient_of,
            "accumulation_node": self.accumulation_node,
            "expected_parts": list(self.expected_parts),
            "actual_parts": list(self.actual_parts),
            "replaced_part": self.replaced_part,
            "replacement_part": self.replacement_part,
        }


@dataclass(frozen=True, slots=True)
class Analysis:
    outcome: Outcome
    reasons: tuple[Reason, ...]
    borrowed_writes: tuple[BorrowedWrite, ...]
    collapse: Collapse | None
    unknown_detail: str | None
    candidate_digest: str
    control_digest: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "reasons": [reason.value for reason in self.reasons],
            "borrowed_writes": [item.to_dict() for item in self.borrowed_writes],
            "collapse": self.collapse.to_dict() if self.collapse else None,
            "unknown_detail": self.unknown_detail,
            "candidate_digest": self.candidate_digest,
            "control_digest": self.control_digest,
        }


def ref(node_id: str) -> dict[str, str]:
    return {"ref": node_id}


def walk_refs(value: Any):
    if isinstance(value, dict):
        if set(value) == {"ref"} and isinstance(value["ref"], str):
            yield value["ref"]
            return
        for child in value.values():
            yield from walk_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_refs(child)


def _closed_dict(raw: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InvalidArtifact(f"{label} must be an object")
    if set(raw) != keys:
        missing = sorted(keys - set(raw))
        extra = sorted(set(raw) - keys)
        raise InvalidArtifact(f"{label} fields mismatch: missing={missing}, extra={extra}")
    return raw


def _string(raw: Any, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(raw, str):
        raise InvalidArtifact(f"{label} must be a string")
    if nonempty and not raw:
        raise InvalidArtifact(f"{label} must be nonempty")
    try:
        encoded = raw.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise InvalidArtifact(f"{label} is not strict UTF-8") from exc
    if len(encoded) > _MAX_STRING_BYTES:
        raise InvalidArtifact(f"{label} exceeds the string byte limit")
    return raw


def _integer(raw: Any, label: str, *, minimum: int | None = None) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise InvalidArtifact(f"{label} must be an integer")
    if minimum is not None and raw < minimum:
        raise InvalidArtifact(f"{label} is below its minimum {minimum}")
    if not _I64[0] <= raw <= _I64[1]:
        raise InvalidArtifact(f"{label} is outside signed-64")
    return raw


def _int_tuple(raw: Any, label: str, *, minimum: int | None = None) -> tuple[int, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in raw):
        raise InvalidArtifact(f"{label} must be an integer list")
    if minimum is not None and any(item < minimum for item in raw):
        raise InvalidArtifact(f"{label} contains a value below its minimum {minimum}")
    if any(not _I64[0] <= item <= _I64[1] for item in raw):
        raise InvalidArtifact(f"{label} contains a value outside signed-64")
    return tuple(raw)


def _string_tuple(raw: Any, label: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise InvalidArtifact(f"{label} must be a string list")
    values = tuple(_string(item, f"{label}[]") for item in raw)
    if len(set(values)) != len(values):
        raise InvalidArtifact(f"{label} contains duplicates")
    return values


def _digest(raw: Any, label: str) -> str:
    value = _string(raw, label)
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise InvalidArtifact(f"{label} must be a lowercase SHA-256 digest")
    return value


def _json_object(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InvalidArtifact(f"{label} must be a JSON object")
    return _json_value(raw, label)


def _json_value(raw: Any, label: str) -> Any:
    if raw is None or isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return _integer(raw, label)
    if isinstance(raw, str):
        return _string(raw, label)
    if isinstance(raw, float):
        if not math.isfinite(raw):
            raise InvalidArtifact(f"{label} contains a non-finite number")
        return raw
    if isinstance(raw, list):
        return [_json_value(item, f"{label}[]") for item in raw]
    if isinstance(raw, dict):
        value = {}
        for key, item in raw.items():
            _string(key, f"{label} object key")
            value[key] = _json_value(item, f"{label}.{key}")
        return value
    raise InvalidArtifact(f"{label} contains non-JSON value {type(raw).__module__}.{type(raw).__qualname__}")


def _fresh_layout_stride(
    shape: tuple[int, ...] | list[int],
    source_shape: tuple[int, ...] | list[int] | None = None,
    source_stride: tuple[int, ...] | list[int] | None = None,
    memory_format: Any = None,
) -> tuple[int, ...] | None:
    """Return only layouts the v2 closed grammar can prove for fresh storage."""
    if any(dimension == 0 for dimension in shape) or (
        source_shape is not None and any(dimension == 0 for dimension in source_shape)
    ):
        return None
    if memory_format is None:
        memory_format = "preserve_format" if source_shape is not None else "contiguous_format"
    elif isinstance(memory_format, dict) and set(memory_format) == {"torch"}:
        memory_format = memory_format["torch"]
    elif not isinstance(memory_format, str):
        return None
    memory_format = memory_format.removeprefix("torch.")
    if memory_format == "contiguous_format":
        stride = []
        step = 1
        for dimension in reversed(shape):
            stride.append(step)
            step *= dimension
        return tuple(reversed(stride))
    if memory_format != "preserve_format" or source_shape is None or source_stride is None:
        return None
    if (
        tuple(shape) != tuple(source_shape)
        or len(source_shape) != len(source_stride)
        or any(dimension <= 1 for dimension in source_shape)
        or any(step < 0 for step in source_stride)
    ):
        return None
    expected = 1
    for step, dimension in sorted(zip(source_stride, source_shape)):
        if step != expected:
            return _fresh_layout_stride(shape, memory_format="contiguous_format")
        expected *= dimension
    return tuple(source_stride)


def _view_result_layout(
    source_shape: tuple[int, ...] | list[int],
    source_stride: tuple[int, ...] | list[int],
    requested: tuple[int, ...] | list[int],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    source_numel = 1
    for dimension in source_shape:
        source_numel *= dimension
        if source_numel > _I64[1]:
            return None
    # The closed grammar models only a literal zero-numel identity; it cannot
    # infer its shape or derive the runtime-specific stride of a reshape.
    if source_numel == 0 and -1 in requested:
        return None
    known_numel = 1
    for dimension in requested:
        if dimension != -1:
            known_numel *= dimension
            if known_numel > _I64[1]:
                return None
    if -1 in requested:
        if not known_numel or source_numel % known_numel:
            return None
        expected = tuple(
            source_numel // known_numel if dimension == -1 else dimension
            for dimension in requested
        )
    elif source_numel != known_numel:
        return None
    else:
        expected = tuple(requested)

    if source_numel == 0:
        if expected == tuple(source_shape):
            return expected, tuple(source_stride)
        return None
    if not source_shape or not any(dimension != 1 for dimension in source_shape):
        return None
    # Tensor view chunks include singleton axes: their strides are observable
    # even when they do not consume elements of the next chunk.
    inferred = [0] * len(expected)
    output_index = len(expected) - 1
    chunk_stride = source_stride[-1]
    source_size = output_size = 1
    for source_index in range(len(source_shape) - 1, -1, -1):
        source_size *= source_shape[source_index]
        joins_previous = (
            source_index > 0
            and (
                source_shape[source_index - 1] == 1
                or source_stride[source_index - 1] == source_size * chunk_stride
            )
        )
        if joins_previous:
            continue
        while output_index >= 0 and (
            output_size < source_size or expected[output_index] == 1
        ):
            stride = output_size * chunk_stride
            if stride > _I64[1]:
                return None
            inferred[output_index] = stride
            output_size *= expected[output_index]
            output_index -= 1
        if output_size != source_size:
            return None
        if source_index:
            chunk_stride = source_stride[source_index - 1]
            source_size = output_size = 1
    if output_index != -1:
        return None
    return expected, tuple(inferred)


_MAX_STRING_BYTES = 1 << 20
_I64 = (-(1 << 63), (1 << 63) - 1)
_MISSING = object()

def _schema_is(value: Any, expected: str) -> bool:
    value_type = type(value)
    if value_type is not str and not issubclass(value_type, str):
        return False
    return str.__eq__(value, expected) is True


def _probe_schema(raw: dict[str, Any]) -> Any:
    for key, value in dict.items(raw):
        if type(key) is str and str.__eq__(key, "schema") is True:
            return value
    return _MISSING
