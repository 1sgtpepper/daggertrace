"""Shared literal graph grammar and state primitives for certificate replay."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .replay_packet import (
    CONFIG_COUNT_MAX,
    MAX_EXTERNAL_FILE_BYTES,
    MAX_SOURCE_BYTES,
    ReplayError,
    _ALIASES,
    _ALLOCATIONS,
    _ALLOCATION_KWARGS,
    _CALL_TARGETS,
    _HOP_WRITE_RESULT_FIELDS,
    _State,
    _STORAGE_ROOT_TARGETS,
    _ALLOCATION_DTYPES,
    _bounded_file,
    _closed,
    _integer,
    _is_digest,
    _sha_bytes,
)
from .replay_source import _function_slice, _host_syntax_error

def _bind_capture_kernels(capture: dict[str, Any], target: dict[str, Any], packet: Path, label: str) -> None:
    bindings = target["kernels"]
    files = {item["packet_path"]: item for item in target["files"]}
    by_key: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = {}
    for binding_index, binding in enumerate(bindings):
        by_key.setdefault(
            (binding["module"], binding["qualname"], binding["source_sha256"]), []
        ).append((binding_index, binding))
    matched: set[int] = set()
    source_bytes = 0
    loaded_sources: dict[str, bytes] = {}
    for kernel in capture["kernels"]:
        candidates = by_key.get(
            (kernel["module"], kernel["qualname"], kernel["source_sha256"]), []
        )
        if len(candidates) != 1:
            raise ReplayError(f"{label} kernel is not bound to one manifest entry")
        binding_index, binding = candidates[0]
        if binding_index in matched:
            raise ReplayError(f"{label} kernel binding is not one-to-one")
        matched.add(binding_index)
        file_path = binding["file"]
        source_path = packet / file_path
        encoded = loaded_sources.get(file_path)
        if encoded is None:
            encoded = _bounded_file(
                source_path,
                MAX_EXTERNAL_FILE_BYTES,
                f"kernel source {file_path}",
                root=packet,
            )
            source_bytes += len(encoded)
            if source_bytes > MAX_SOURCE_BYTES:
                raise ReplayError(f"{label} source bindings exceed their aggregate byte limit")
            loaded_sources[file_path] = encoded
        if len(encoded) != files[file_path]["size"] or _sha_bytes(encoded) != files[file_path]["sha256"]:
            raise ReplayError(f"kernel source file {file_path} does not match its manifest bytes")
        try:
            source = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReplayError(f"kernel source file {file_path} is not UTF-8") from exc
        try:
            function_source = _function_slice(source, kernel["qualname"], bounded=True)
        except SyntaxError as exc:
            raise _host_syntax_error(f"kernel {kernel['qualname']} source", exc) from exc
        except ValueError as exc:
            raise ReplayError(
                f"kernel {kernel['qualname']} source slice is not manifest-bound"
            ) from exc
        if function_source != kernel["source"] or _sha_bytes(function_source.encode("utf-8")) != binding["source_sha256"]:
            raise ReplayError(f"kernel {kernel['qualname']} source slice is not manifest-bound")
    if matched != set(range(len(bindings))):
        raise ReplayError(f"{label} does not provide total bidirectional kernel binding")


def _graph(raw: Any, label: str, expected_kind: str) -> None:
    graph = _closed(raw, {"kind", "nodes", "output_roles"}, label)
    if graph["kind"] != expected_kind:
        raise ReplayError(f"{label}.kind changed")
    if not isinstance(graph["nodes"], list) or not graph["nodes"] or len(graph["nodes"]) > 256:
        raise ReplayError(f"{label}.nodes must be nonempty")
    output_nodes = 0
    node_ids: set[str] = set()
    allowed_placeholder_roles = {"primal", "tangent"}
    if expected_kind == "backward":
        allowed_placeholder_roles.add("saved_output")
    for index, raw_node in enumerate(graph["nodes"]):
        node = _closed(
            raw_node,
            {"id", "op", "target", "args", "kwargs", "role", "tensor", "gradient_accumulation"},
            f"{label}.node",
        )
        if not isinstance(node["id"], str) or node["id"] != f"n{index}":
            raise ReplayError(f"{label} node ids must be dense topological n0..nN")
        if node["id"] in node_ids:
            raise ReplayError(f"{label} node ids are not unique")
        node_ids.add(node["id"])
        if not isinstance(node["op"], str) or not isinstance(node["target"], str):
            raise ReplayError(f"{label}.{node['id']} op/target is malformed")
        if node["op"] == "placeholder":
            if node["target"] != "placeholder" or node["args"] != {"tuple": []} or node["kwargs"] != {}:
                raise ReplayError(f"{label}.{node['id']} placeholder grammar changed")
            if node["role"] is not None:
                kind, _ = _role_descriptor(node["role"], f"{label}.{node['id']}.role")
                if kind not in allowed_placeholder_roles:
                    raise ReplayError(f"{label}.{node['id']} placeholder role changed")
        elif node["op"] == "output":
            if node["target"] != "output" or node["kwargs"] != {}:
                raise ReplayError(f"{label}.{node['id']} output grammar changed")
            output_nodes += 1
        elif node["op"] != "call_function" or node["target"] not in _CALL_TARGETS:
            raise ReplayError(
                f"unsupported replay node {label}.{node['id']}: {node['op']}/{node['target']}"
            )
        elif node["target"] == "triton.functional":
            if node["args"] != []:
                raise ReplayError(f"{label}.{node['id']} HOP positional arguments changed")
        else:
            _v2_call_shape(node, graph["nodes"], index, f"{label}.{node['id']}")
        if node["op"] != "placeholder" and node["role"] is not None:
            raise ReplayError(f"{label}.{node['id']} has a role outside a placeholder")
        if node["op"] == "placeholder" and node["tensor"] is None:
            raise ReplayError(f"{label}.{node['id']} placeholder lacks tensor metadata")
        if node["tensor"] is not None:
            _tensor(node["tensor"], f"{label}.{node['id']}.tensor")
        if node["op"] == "placeholder" or node["op"] == "call_function":
            expects_tensor = node["target"] != "triton.functional"
            if expects_tensor != (node["tensor"] is not None):
                raise ReplayError(f"{label}.{node['id']} tensor-result metadata changed")
        elif node["tensor"] is not None:
            raise ReplayError(f"{label}.{node['id']} output unexpectedly has tensor metadata")
        if not isinstance(node["gradient_accumulation"], bool):
            raise ReplayError(f"{label}.{node['id']} gradient marker is malformed")
        if node["gradient_accumulation"] and node["target"] != "aten.add":
            raise ReplayError(f"{label}.{node['id']} has a gradient marker on a non-add")
        for reference in _refs(node["args"]):
            if reference not in node_ids or reference == node["id"]:
                raise ReplayError(f"{label}.{node['id']} has an invalid reference {reference}")
        for reference in _refs(node["kwargs"]):
            if reference not in node_ids or reference == node["id"]:
                raise ReplayError(f"{label}.{node['id']} has an invalid reference {reference}")
        prior_ids = {item["id"] for item in graph["nodes"][:index]}
        if any(reference not in prior_ids for reference in _refs(node["args"])):
            raise ReplayError(f"{label}.{node['id']} references a later node")
        if any(reference not in prior_ids for reference in _refs(node["kwargs"])):
            raise ReplayError(f"{label}.{node['id']} references a later node")
    if output_nodes != 1 or graph["nodes"][-1]["op"] != "output" or graph["nodes"][-1]["target"] != "output":
        raise ReplayError(f"{label} does not end in output")
    if not isinstance(graph["output_roles"], list):
        raise ReplayError(f"{label}.output_roles must be a list")
    allowed = {"output", "saved_output"} if expected_kind == "forward" else {"gradient"}
    role_paths = []
    for index, role in enumerate(graph["output_roles"]):
        descriptor = _role_descriptor(role, f"{label}.output_roles[{index}]")
        if descriptor[0] not in allowed:
            raise ReplayError(f"{label}.output_roles[{index}] has the wrong role kind")
        role_paths.append(descriptor)
    if len(role_paths) != len(set(role_paths)):
        raise ReplayError(f"{label}.output_roles contains a duplicate role path")
    outputs = _output_refs(graph["nodes"][-1]["args"])
    if len(outputs) != len(graph["output_roles"]):
        raise ReplayError(f"{label} output values/roles differ")
    prior = {item["id"]: item for item in graph["nodes"][:-1]}
    for value in outputs:
        reference = _reference(value)
        if reference not in prior or prior[reference]["tensor"] is None:
            raise ReplayError(f"{label} output references a non-tensor node")


def _tensor(raw: Any, label: str) -> None:
    tensor = _closed(raw, {"shape", "stride", "dtype", "device", "storage_offset", "alias_group"}, label)
    if not isinstance(tensor["shape"], list) or not isinstance(tensor["stride"], list):
        raise ReplayError(f"{label} is malformed")
    if len(tensor["shape"]) != len(tensor["stride"]) or len(tensor["shape"]) > 64:
        raise ReplayError(f"{label} is malformed")
    for index, value in enumerate(tensor["shape"]):
        _integer(value, f"{label}.shape[{index}]", minimum=0, maximum=(1 << 63) - 1)
    for index, value in enumerate(tensor["stride"]):
        _integer(
            value,
            f"{label}.stride[{index}]",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
    elements = 1
    for dimension in tensor["shape"]:
        elements *= dimension
        if elements > (1 << 63) - 1:
            raise ReplayError(f"{label}.shape product exceeds signed-64")
    _integer(
        tensor["storage_offset"],
        f"{label}.storage_offset",
        minimum=0,
        maximum=(1 << 63) - 1,
    )
    if (
        not isinstance(tensor["dtype"], str)
        or not tensor["dtype"]
        or not isinstance(tensor["device"], str)
        or not tensor["device"]
        or (tensor["alias_group"] is not None and (not isinstance(tensor["alias_group"], str) or not tensor["alias_group"]))
    ):
        raise ReplayError(f"{label} is malformed")
def _role_descriptor(raw: Any, label: str) -> tuple[str, int]:
    if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
        raise ReplayError(f"{label} is malformed")
    kind = raw["kind"]
    if kind in {"primal", "output", "saved_output"}:
        role = _closed(raw, {"kind", "index"}, label)
        return kind, _integer(role["index"], f"{label}.index", minimum=0)
    if kind == "tangent":
        role = _closed(raw, {"kind", "output"}, label)
        output_kind, index = _role_descriptor(role["output"], f"{label}.output")
        if output_kind != "output":
            raise ReplayError(f"{label} does not identify a forward output")
        return kind, index
    if kind == "gradient":
        role = _closed(raw, {"kind", "primal"}, label)
        primal_kind, index = _role_descriptor(role["primal"], f"{label}.primal")
        if primal_kind != "primal":
            raise ReplayError(f"{label} does not identify a forward primal")
        return kind, index
    raise ReplayError(f"{label} has unsupported role kind {kind!r}")


def _runtime_tensor(raw: Any, label: str) -> None:
    tensor = _closed(raw, {"shape", "stride", "storage_offset", "dtype", "device", "requires_grad"}, label)
    if (
        not isinstance(tensor["shape"], list)
        or not isinstance(tensor["stride"], list)
        or len(tensor["shape"]) != len(tensor["stride"])
        or len(tensor["shape"]) > 64
    ):
        raise ReplayError(f"{label} is malformed")
    for index, value in enumerate(tensor["shape"]):
        _integer(value, f"{label}.shape[{index}]", minimum=0, maximum=(1 << 63) - 1)
    for index, value in enumerate(tensor["stride"]):
        _integer(
            value,
            f"{label}.stride[{index}]",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
    elements = 1
    for dimension in tensor["shape"]:
        elements *= dimension
        if elements > (1 << 63) - 1:
            raise ReplayError(f"{label}.shape product exceeds signed-64")
    _integer(
        tensor["storage_offset"],
        f"{label}.storage_offset",
        minimum=0,
        maximum=(1 << 63) - 1,
    )
    if not isinstance(tensor["dtype"], str) or not tensor["dtype"] or not isinstance(tensor["device"], str) or not tensor["device"] or not isinstance(tensor["requires_grad"], bool):
        raise ReplayError(f"{label} is malformed")
def _kernel_runtime_record(raw: Any) -> None:
    runtime = _closed(raw, {"wrapper_type", "leaf_type", "config", "jit_cache_key"}, "kernel.runtime")
    if (
        runtime["wrapper_type"] != "triton.runtime.autotuner.Autotuner"
        or runtime["leaf_type"] != "triton.runtime.jit.JITFunction"
        or not _is_digest(runtime["jit_cache_key"])
    ):
        raise ReplayError("kernel runtime wrapper identity is outside the versioned profile")
    config = _closed(runtime["config"], {"kwargs", "num_warps", "num_ctas", "num_stages", "maxnreg", "pre_hook", "ir_override"}, "kernel.runtime.config")
    if not isinstance(config["kwargs"], dict) or config["kwargs"] != {}:
        raise ReplayError("kernel runtime config kwargs are malformed")
    for key in ("num_warps", "num_ctas", "num_stages"):
        _integer(config[key], f"kernel.runtime.config.{key}", minimum=1, maximum=CONFIG_COUNT_MAX)
    if config["num_warps"] & (config["num_warps"] - 1):
        raise ReplayError("kernel.runtime.config.num_warps must be a positive power of two")
    if (
        config["maxnreg"] is not None
        or config["pre_hook"] is not None
        or config["ir_override"] is not None
    ):
        raise ReplayError("kernel runtime hooks and IR overrides are outside the profile")


def _kernel_map(capture: dict[str, Any]) -> dict[str, Any]:
    return {kernel["id"]: kernel for kernel in capture["kernels"]}


def _validate_storage_identities(nodes: list[dict[str, Any]]) -> None:
    nodes_by_id = {node["id"]: node for node in nodes}
    root_groups: dict[str, str] = {}
    for node in nodes:
        node_id = node["id"]
        target = node["target"]
        if node["op"] == "output" or target == "triton.functional":
            continue
        if node["op"] == "placeholder" or target in _STORAGE_ROOT_TARGETS:
            group = _captured_alias_group(node, node_id)
            prior = root_groups.get(group)
            if prior is not None:
                if node["op"] == "placeholder" and nodes_by_id[prior]["op"] == "placeholder":
                    raise ReplayError(f"distinct placeholders {prior}/{node_id} share storage")
                raise ReplayError(f"fresh result {node_id} reuses storage from {prior}")
            root_groups[group] = node_id
            continue
        if target not in _ALIASES:
            continue
        group = _captured_alias_group(node, node_id)
        references = list(_refs(node["args"]))
        if len(references) != 1 or references[0] not in nodes_by_id:
            raise ReplayError(f"alias node {node_id} does not have one referenced input")
        source = nodes_by_id[references[0]]
        source_group = _captured_alias_group(source, references[0])
        if group != source_group:
            raise ReplayError(f"alias node {node_id} captured storage identity differs from input {references[0]}")


def _captured_alias_group(node: dict[str, Any], node_id: str) -> str:
    tensor = node.get("tensor")
    if not isinstance(tensor, dict):
        raise ReplayError(f"{node_id} lacks tensor metadata for captured storage identity")
    group = tensor.get("alias_group")
    if not isinstance(group, str) or not group:
        if node["op"] == "placeholder":
            raise ReplayError(f"placeholder {node_id} lacks captured storage identity")
        if node["target"] in _ALIASES:
            raise ReplayError(f"view node {node_id} lacks captured storage identity")
        raise ReplayError(f"fresh result {node_id} lacks captured storage identity")
    return group


def _backward_twin_skeleton(capture: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the observable backward structure used by paired replay.

    View-like wrappers and fresh destination allocations can differ between
    AOTAutograd lanes and are already modeled by the ownership transition
    system.  Other reachable operators stay in the skeleton; this prevents a
    connected computation inserted into one lane from being hidden by the
    contribution multiset relation.
    """

    graph = capture["backward"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    reachable: set[str] = set()
    pending = list(_refs(graph["nodes"][-1]["args"]))
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            continue
        node = nodes[node_id]
        reachable.add(node_id)
        pending.extend(_refs(node["args"]))
        pending.extend(_refs(node["kwargs"]))
    skeleton: list[dict[str, Any]] = []
    for node in graph["nodes"]:
        if node["id"] not in reachable:
            continue
        if node["target"] in _ALIASES | _ALLOCATIONS:
            continue
        tensor = None if node["tensor"] is None else dict(node["tensor"])
        if tensor is not None:
            tensor.pop("alias_group")
        skeleton.append(
            {
                "op": node["op"],
                "target": node["target"],
                "args": _without_refs(node["args"]),
                "kwargs": _without_refs(node["kwargs"]),
                "role": node["role"],
                "tensor": tensor,
                "gradient_accumulation": node["gradient_accumulation"],
                "output_roles": (
                    list(graph["output_roles"])
                    if node["op"] == "output"
                    else None
                ),
            }
        )
    return skeleton


def _single(value, states):
    dependencies = _states(value, states)
    if len(dependencies) != 1:
        raise ReplayError("alias op does not have exactly one tensor input")
    return dependencies[0]


def _states(value, states):
    result = []
    if isinstance(value, dict):
        if set(value) == {"ref"}:
            if value["ref"] in states:
                result.append(states[value["ref"]])
        else:
            for child in value.values():
                result.extend(_states(child, states))
    elif isinstance(value, list):
        for child in value:
            result.extend(_states(child, states))
    return result


def _state(value, states):
    reference = _reference(value)
    if reference not in states:
        raise ReplayError(f"reference {reference} is not a tensor state")
    return states[reference]


def _reference(value):
    if not isinstance(value, dict) or set(value) != {"ref"} or not isinstance(value["ref"], str):
        raise ReplayError("expected a node reference")
    return value["ref"]


def _validate_fresh_result_metadata(
    node: dict[str, Any], shape: list[int], dtype: str, device: str, label: str
) -> None:
    result = _closed(
        node["tensor"],
        {"shape", "stride", "dtype", "device", "storage_offset", "alias_group"},
        f"{label}.tensor",
    )
    if any(result[key] != expected for key, expected in (
        ("shape", shape), ("dtype", dtype), ("device", device), ("storage_offset", 0)
    )):
        raise ReplayError(f"{label} result metadata changed")
    if result["stride"] != _fresh_stride(None, shape, {}, label):
        raise ReplayError(f"{label} result layout changed")


def _validate_getitem_result_metadata(
    node: dict[str, Any], hop: dict[str, Any], key: str, prior: dict[str, dict[str, Any]], label: str
) -> None:
    kwargs = hop["kwargs"]
    if type(kwargs) is not dict or type(kwargs.get("bindings")) is not dict or type(kwargs.get("writes")) is not list:
        raise ReplayError(f"{label} getitem grammar changed")
    if key not in kwargs["writes"] or key not in kwargs["bindings"]:
        raise ReplayError(f"{label} does not select a declared HOP write")
    binding = prior.get(_reference(kwargs["bindings"][key]))
    if binding is None or binding["tensor"] is None:
        raise ReplayError(f"{label} getitem write binding is malformed")
    result = _closed(
        node["tensor"],
        {"shape", "stride", "dtype", "device", "storage_offset", "alias_group"},
        f"{label}.tensor",
    )
    if any(result[field] != binding["tensor"][field] for field in _HOP_WRITE_RESULT_FIELDS):
        raise ReplayError(f"{label} getitem result metadata changed")


def _tuple(value):
    if not isinstance(value, dict) or set(value) != {"tuple"} or not isinstance(value["tuple"], list):
        raise ReplayError("expected encoded tuple")
    return value["tuple"]


def _v2_call_shape(node: dict[str, Any], nodes: list[dict[str, Any]], index: int, label: str) -> None:
    args, kwargs = _tuple(node["args"]), node["kwargs"]
    if type(kwargs) is not dict:
        raise ReplayError(f"{label}.kwargs is malformed")
    prior = {item["id"]: item for item in nodes[:index]}

    def tensor_ref(value: Any) -> dict[str, Any]:
        reference = _reference(value)
        item = prior.get(reference)
        if item is None or item["tensor"] is None:
            raise ReplayError(f"{label} requires a prior tensor reference")
        return item

    target = node["target"]
    if target in {"aten.view", "aten._unsafe_view"}:
        if kwargs or len(args) != 2 or type(args[1]) is not list or not 0 < len(args[1]) <= 64:
            raise ReplayError(f"{label} view grammar changed")
        if sum(item == -1 for item in args[1]) > 1 or any(
            type(item) is not int or item < -1 or item > (1 << 63) - 1 for item in args[1]
        ):
            raise ReplayError(f"{label} view grammar changed")
        source = tensor_ref(args[0])
        _validate_view_result_metadata(node, source, args[1], label)
    elif target in _ALIASES | {"aten.clone"}:
        if kwargs or len(args) != 1:
            raise ReplayError(f"{label} unary grammar changed")
        source = tensor_ref(args[0])
        result = _closed(
            node["tensor"],
            {"shape", "stride", "dtype", "device", "storage_offset", "alias_group"},
            f"{label}.tensor",
        )
        if target in _ALIASES and any(
            result[key] != source["tensor"][key]
            for key in ("shape", "stride", "alias_group", "dtype", "device", "storage_offset")
        ):
            raise ReplayError(f"{label} alias layout changed")
        if target == "aten.clone" and any(
            result[key] != source["tensor"][key]
            for key in ("shape", "dtype", "device")
        ):
            raise ReplayError(f"{label} clone layout changed")
        if (
            target == "aten.clone"
            and (
                result["storage_offset"] != 0
                or result["stride"]
                != _fresh_stride(source["tensor"], source["tensor"]["shape"], {}, label)
            )
        ):
            raise ReplayError(f"{label} clone result layout changed")
    elif target in _ALLOCATIONS:
        _validate_allocation_grammar(node, nodes, index, label)
    elif target in {"aten.mul", "aten.add"}:
        if kwargs or len(args) != 2:
            raise ReplayError(f"{label} {target.rsplit('.', 1)[-1]} grammar changed")
        sources = []
        for value in args:
            if type(value) is dict and set(value) == {"ref"}:
                sources.append(tensor_ref(value))
            elif target == "aten.mul" and type(value) is int:
                _integer(value, f"{label}.operand", minimum=-(1 << 63), maximum=(1 << 63) - 1)
            elif target == "aten.mul" and type(value) is float and math.isfinite(value):
                continue
            else:
                raise ReplayError(f"{label} {target.rsplit('.', 1)[-1]} grammar changed")
        if not sources or any(
            source["tensor"][field] != sources[0]["tensor"][field]
            for source in sources[1:]
            for field in ("shape", "dtype", "device")
        ):
            raise ReplayError(f"{label} {target.rsplit('.', 1)[-1]} grammar changed")
        source = sources[0]["tensor"]
        _validate_fresh_result_metadata(node, source["shape"], source["dtype"], source["device"], label)
    elif target == "aten.sum":
        if kwargs or len(args) != 2 or type(args[1]) is not list or len(args[1]) > 64:
            raise ReplayError(f"{label} sum grammar changed")
        source = tensor_ref(args[0])
        dimensions = [
            _integer(item, f"{label}.dimension", minimum=-(1 << 63), maximum=(1 << 63) - 1)
            for item in args[1]
        ]
        rank = len(source["tensor"]["shape"])
        if (
            (not rank and dimensions)
            or (
                rank
                and (
                    any(not -rank <= item < rank for item in dimensions)
                    or len({item % rank for item in dimensions}) != len(dimensions)
                )
            )
        ):
            raise ReplayError(f"{label} sum grammar changed")
        reduced = {dimension % rank for dimension in dimensions} if rank else set()
        _validate_fresh_result_metadata(
            node,
            [size for position, size in enumerate(source["tensor"]["shape"]) if position not in reduced],
            source["tensor"]["dtype"],
            source["tensor"]["device"],
            label,
        )
    elif target == "operator.getitem":
        if kwargs or len(args) != 2 or type(args[1]) is not str or not args[1]:
            raise ReplayError(f"{label} getitem grammar changed")
        reference = _reference(args[0])
        hop = prior.get(reference)
        if hop is None or hop.get("target") != "triton.functional":
            raise ReplayError(f"{label} getitem grammar changed")
        _validate_getitem_result_metadata(node, hop, args[1], prior, label)
    else:
        raise ReplayError(f"{label} target grammar changed")


def _validate_allocation_grammar(node: dict[str, Any], nodes: list[dict[str, Any]], index: int, label: str) -> None:
    values = _tuple(node["args"])
    if len(values) != 1:
        raise ReplayError(f"{label} requires exactly one positional argument")
    kwargs = node["kwargs"]
    if not isinstance(kwargs, dict) or not all(isinstance(key, str) and key in _ALLOCATION_KWARGS for key in kwargs):
        raise ReplayError(f"{label} has unsupported keyword arguments")
    if _contains_encoded_ref(kwargs):
        raise ReplayError(f"{label} keyword arguments contain tensor refs")
    for key, value in kwargs.items():
        if key == "pin_memory":
            if type(value) is not bool:
                raise ReplayError(f"{label} allocation keyword {key} is malformed")
            continue
        if type(value) is dict and set(value) == {"torch"}:
            token = value["torch"]
        elif key == "device" and type(value) is str:
            # Older v2 callers may spell the device literal directly; the
            # producer's normalized form is the encoded torch token above.
            token = value
        else:
            raise ReplayError(f"{label} allocation keyword {key} is malformed")
        if type(token) is not str or not token or len(token.encode("utf-8")) > 128:
            raise ReplayError(f"{label} allocation keyword {key} is malformed")
        normalized = token.removeprefix("torch.")
        if (
            key == "dtype" and normalized not in _ALLOCATION_DTYPES
        ) or (
            key == "layout" and normalized != "strided"
        ) or (
            key == "memory_format"
            and normalized not in {"contiguous_format", "preserve_format"}
        ):
            raise ReplayError(f"{label} allocation keyword {key} is unsupported")
        if key == "device" and token.startswith("cuda:"):
            suffix = token[5:]
            if not suffix.isascii() or not suffix.isdigit() or str(int(suffix)) != suffix or int(suffix) > 65535:
                raise ReplayError(f"{label} allocation keyword {key} is malformed")
    argument = values[0]
    if node["target"] in {"aten.empty", "aten.zeros"}:
        if not {"dtype", "device"} <= set(kwargs):
            raise ReplayError(f"{label} allocation must declare dtype and device")
        if type(argument) is not list or len(argument) > 64:
            raise ReplayError(f"{label} allocation size changed")
        for dimension in argument:
            _integer(dimension, f"{label}.size", minimum=0, maximum=(1 << 63) - 1)
        elements = 1
        for dimension in argument:
            elements *= dimension
            if elements > (1 << 63) - 1:
                raise ReplayError(f"{label} allocation size product exceeds signed-64")
        result = _closed(
            node["tensor"],
            {"shape", "stride", "dtype", "device", "storage_offset", "alias_group"},
            f"{label}.tensor",
        )
        dtype = kwargs["dtype"]["torch"].removeprefix("torch.")
        device = kwargs["device"]
        device = device["torch"] if isinstance(device, dict) else device
        if any(
            result[key] != expected
            for key, expected in (("shape", argument), ("dtype", dtype), ("device", device))
        ):
            raise ReplayError(f"{label} allocation result metadata changed")
        if (
            result["storage_offset"] != 0
            or result["stride"] != _fresh_stride(None, argument, kwargs, label)
        ):
            raise ReplayError(f"{label} allocation result layout changed")
        return
    reference = _reference(argument)
    nodes_by_id = {item["id"]: item for item in nodes}
    if reference not in nodes_by_id or reference not in {item["id"] for item in nodes[:index]} or nodes_by_id[reference]["tensor"] is None:
        raise ReplayError(f"{label} requires one prior tensor reference")
    if node["target"] in {"aten.empty_like", "aten.zeros_like"}:
        source = nodes_by_id[reference]
        result = _closed(
            node["tensor"],
            {"shape", "stride", "dtype", "device", "storage_offset", "alias_group"},
            f"{label}.tensor",
        )
        expected_dtype = source["tensor"]["dtype"]
        expected_device = source["tensor"]["device"]
        for key in ("dtype", "device"):
            if key not in kwargs:
                continue
            override = kwargs[key]
            if isinstance(override, dict):
                override = override["torch"]
            if key == "dtype":
                override = override.removeprefix("torch.")
                expected_dtype = override
            else:
                expected_device = override
        if any(
            result[key] != expected
            for key, expected in (
                ("shape", source["tensor"]["shape"]),
                ("dtype", expected_dtype),
                ("device", expected_device),
            )
        ):
            raise ReplayError(f"{label} like result metadata changed")
        if (
            result["storage_offset"] != 0
            or result["stride"]
            != _fresh_stride(source["tensor"], source["tensor"]["shape"], kwargs, label)
        ):
            raise ReplayError(f"{label} like result layout changed")


def _contiguous_stride(shape: list[int]) -> list[int]:
    stride = []
    step = 1
    for dimension in reversed(shape):
        stride.append(step)
        step *= dimension
    return list(reversed(stride))


def _preserved_stride(
    tensor: dict[str, Any], shape: list[int], label: str
) -> list[int]:
    if (
        tensor["shape"] != shape
        or any(dimension <= 1 for dimension in tensor["shape"])
        or any(step < 0 for step in tensor["stride"])
    ):
        raise ReplayError(f"{label} layout is not provable")
    expected = 1
    for step, dimension in sorted(zip(tensor["stride"], tensor["shape"])):
        if step != expected:
            return _contiguous_stride(tensor["shape"])
        expected *= dimension
    return tensor["stride"]


def _fresh_stride(
    source: dict[str, Any] | None,
    shape: list[int],
    kwargs: dict[str, Any],
    label: str,
) -> list[int]:
    if any(dimension == 0 for dimension in shape) or (
        source is not None and any(dimension == 0 for dimension in source["shape"])
    ):
        raise ReplayError(f"{label} layout is not provable")
    value = kwargs.get("memory_format")
    memory_format = "preserve_format" if source is not None else "contiguous_format"
    if value is not None:
        value = value["torch"] if isinstance(value, dict) else value
        memory_format = value.removeprefix("torch.")
    if memory_format == "contiguous_format":
        return _contiguous_stride(shape)
    if memory_format == "preserve_format" and source is not None:
        return _preserved_stride(source, shape, label)
    raise ReplayError(f"{label} layout is not provable")


def _allocation_signature(node: dict[str, Any], states: dict[str, _State], capture_digest: str) -> dict[str, Any]:
    tensor = node["tensor"]
    if not isinstance(tensor, dict):
        raise ReplayError(f"allocation node {node['id']} lacks tensor metadata")
    dependencies = []
    for reference in _refs(node["args"]):
        if reference not in states:
            raise ReplayError(f"allocation node {node['id']} references a non-tensor state")
        state = states[reference]
        dependencies.append({"node": reference, "buffer": state.buffer, "version": state.version, "parts": list(state.parts)})
    layout = dict(tensor)
    layout.pop("alias_group")
    return {
        "capture_digest": capture_digest,
        "target": node["target"],
        "args": {"dependencies": dependencies, "literals": _without_refs(node["args"])},
        "kwargs": node["kwargs"],
        "layout": layout,
        "node": node["id"],
    }


def _contains_encoded_ref(value: Any) -> bool:
    if isinstance(value, dict):
        if "ref" in value:
            return True
        return any(_contains_encoded_ref(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_encoded_ref(child) for child in value)
    return False


def _binding_layout(nodes: dict[str, dict[str, Any]], value: Any) -> dict[str, Any]:
    node = nodes[_reference(value)]
    if node["tensor"] is None:
        raise ReplayError("HOP binding lacks tensor metadata")
    layout = dict(node["tensor"])
    layout.pop("alias_group")
    return layout


def _output_refs(value):
    outer = _tuple(value)
    if len(outer) != 1:
        raise ReplayError("output node aggregate changed")
    return _tuple(outer[0])


def _without_refs(value):
    if isinstance(value, dict):
        if set(value) == {"ref"}:
            return "<tensor>"
        return {key: _without_refs(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_without_refs(child) for child in value]
    return value


def _refs(value):
    if isinstance(value, dict):
        if set(value) == {"ref"}:
            yield value["ref"]
        else:
            for child in value.values():
                yield from _refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _refs(child)


def _validate_view_result_metadata(
    node: dict[str, Any], source: dict[str, Any], requested: list[int], label: str
) -> None:
    result = node["tensor"]
    source_tensor = source["tensor"]
    if not isinstance(result, dict):
        raise ReplayError(f"{label} view result metadata is missing")
    _closed(
        result,
        {"shape", "stride", "dtype", "device", "storage_offset", "alias_group"},
        f"{label}.tensor",
    )
    for key in ("alias_group", "dtype", "device", "storage_offset"):
        if result[key] != source_tensor[key]:
            raise ReplayError(f"{label} view result {key} changed")

    # A view retains the source storage and consumes exactly the source number
    # of elements; a single -1 is resolved from that count.
    source_numel = 1
    for dimension in source_tensor["shape"]:
        source_numel *= dimension
        if source_numel > (1 << 63) - 1:
            raise ReplayError(f"{label} source shape product exceeds signed-64")
    if source_numel == 0 and -1 in requested:
        raise ReplayError(f"{label} view result stride changed")
    known_numel = 1
    for dimension in requested:
        if dimension == -1:
            continue
        known_numel *= dimension
        if known_numel > (1 << 63) - 1:
            raise ReplayError(f"{label} view shape product exceeds signed-64")
    if -1 in requested:
        if known_numel == 0 or source_numel % known_numel:
            raise ReplayError(f"{label} view result shape is incompatible")
        inferred = source_numel // known_numel
        expected = [inferred if dimension == -1 else dimension for dimension in requested]
    else:
        if source_numel != known_numel:
            raise ReplayError(f"{label} view result shape is incompatible")
        expected = requested
    if result["shape"] != expected:
        raise ReplayError(f"{label} view result shape changed")

    # Independently preserve the model's closed zero-numel language: only a
    # literal identity has a modeled stride.
    if source_numel == 0:
        if expected != source_tensor["shape"]:
            raise ReplayError(f"{label} view result stride changed")
        expected_stride = source_tensor["stride"]
    elif not source_tensor["shape"] or not any(
        dimension != 1 for dimension in source_tensor["shape"]
    ):
        raise ReplayError(f"{label} view result stride changed")
    else:
        expected_stride = [0] * len(expected)
        output_index = len(expected) - 1
        chunk_stride = source_tensor["stride"][-1]
        source_size = output_size = 1
        for source_index in range(len(source_tensor["shape"]) - 1, -1, -1):
            source_size *= source_tensor["shape"][source_index]
            joins_previous = (
                source_index > 0
                and (
                    source_tensor["shape"][source_index - 1] == 1
                    or source_tensor["stride"][source_index - 1]
                    == source_size * chunk_stride
                )
            )
            if joins_previous:
                continue
            while output_index >= 0 and (
                output_size < source_size or expected[output_index] == 1
            ):
                stride = output_size * chunk_stride
                if stride > (1 << 63) - 1:
                    raise ReplayError(f"{label} view result stride changed")
                expected_stride[output_index] = stride
                output_size *= expected[output_index]
                output_index -= 1
            if output_size != source_size:
                raise ReplayError(f"{label} view result stride changed")
            if source_index:
                chunk_stride = source_tensor["stride"][source_index - 1]
                source_size = output_size = 1
        if output_index != -1:
            raise ReplayError(f"{label} view result stride changed")
    if result["stride"] != expected_stride:
        raise ReplayError(f"{label} view result stride changed")
