"""Version 2 capture grammar and literal ownership execution for replay."""

from __future__ import annotations

from collections import Counter
from typing import Any


from .replay_packet import (
    CAPTURE_SCHEMA,
    EXPECTED_CAPTURE_CONFIG,
    MAX_DERIVED_PARTS,
    MAX_KERNELS,
    MAX_STRING_BYTES,
    ReplayError,
    SUPPORTED_TORCH,
    SUPPORTED_TORCH_GIT,
    SUPPORTED_TRITON,
    _ALIASES,
    _ALLOCATIONS,
    _HOP_WRITE_RESULT_FIELDS,
    _PURE,
    _State,
    _canonical,
    _closed,
    _integer,
    _is_digest,
    _sha,
    _sha_bytes,
)
from .replay_source import _source_effects
from .replay_graph import (
    _allocation_signature,
    _backward_twin_skeleton,
    _binding_layout,
    _contains_encoded_ref,
    _graph,
    _kernel_map,
    _kernel_runtime_record,
    _output_refs,
    _reference,
    _role_descriptor,
    _runtime_tensor,
    _single,
    _state,
    _states,
    _tuple,
    _validate_allocation_grammar,
    _validate_storage_identities,
    _without_refs,
)


def _capture(raw: Any, label: str) -> dict[str, Any]:
    capture = _closed(
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
        label,
    )
    if capture["schema"] != CAPTURE_SCHEMA:
        raise ReplayError(f"{label} capture schema changed")
    if not isinstance(capture["label"], str) or not capture["label"]:
        raise ReplayError(f"{label} capture label is malformed")
    framework = _closed(
        capture["framework"],
        {"python", "torch", "torch_git", "triton", "cuda_runtime"},
        f"{label}.framework",
    )
    if (
        framework["torch"] != SUPPORTED_TORCH
        or framework["torch_git"] != SUPPORTED_TORCH_GIT
        or framework["triton"] != SUPPORTED_TRITON
    ):
        raise ReplayError(f"{label} framework is outside the versioned profile")
    for key in ("python", "torch_git", "cuda_runtime"):
        if not isinstance(framework[key], str) or not framework[key]:
            raise ReplayError(f"{label}.framework.{key} is malformed")
    if _canonical(capture["capture_config"]) != _canonical(EXPECTED_CAPTURE_CONFIG):
        raise ReplayError(f"{label} capture configuration changed")
    _graph(capture["forward"], f"{label}.forward", "forward")
    _graph(capture["backward"], f"{label}.backward", "backward")
    kernels = capture["kernels"]
    if not isinstance(kernels, list) or not kernels or len(kernels) > MAX_KERNELS:
        raise ReplayError(f"{label} has no source-bound kernels")
    kernel_ids: set[str] = set()
    for kernel in kernels:
        _kernel(kernel)
        if kernel["id"] in kernel_ids:
            raise ReplayError(f"{label} has duplicate kernel ids")
        kernel_ids.add(kernel["id"])
    referenced = {
        node["kwargs"]["kernel"]
        for graph in (capture["forward"], capture["backward"])
        for node in graph["nodes"]
        if node["target"] == "triton.functional"
    }
    if referenced != kernel_ids:
        raise ReplayError(f"{label} kernel table is not referentially total")
    _validate_hops(capture, label)
    _validate_storage_identities(capture["forward"]["nodes"])
    _validate_storage_identities(capture["backward"]["nodes"])
    input_slots, required_gradients = _capture_diagnostics(capture["diagnostics"], label)
    _validate_role_domains(capture, label, input_slots, required_gradients)
    return capture


def _validate_role_domains(
    capture: dict[str, Any],
    label: str,
    input_slots: set[int],
    required_gradients: set[int],
) -> None:
    forward_primals: set[int] = set()
    for node in capture["forward"]["nodes"]:
        if node["op"] != "placeholder":
            continue
        if node["role"] is None:
            raise ReplayError(f"{label}.forward placeholder {node['id']} lacks a primal role")
        kind, index = _role_descriptor(node["role"], f"{label}.forward.{node['id']}.role")
        if kind != "primal" or index in forward_primals:
            raise ReplayError(f"{label}.forward placeholder roles are not total and unique")
        forward_primals.add(index)
    if forward_primals != input_slots:
        raise ReplayError(
            f"{label} forward primal roles {sorted(forward_primals)} do not match "
            f"runtime inputs {sorted(input_slots)}"
        )
    forward_outputs = {
        _role_descriptor(role, f"{label}.forward.output_role")[1]
        for role in capture["forward"]["output_roles"]
        if role["kind"] == "output"
    }
    forward_saved_outputs = {
        _role_descriptor(role, f"{label}.forward.output_role")[1]
        for role in capture["forward"]["output_roles"]
        if role["kind"] == "saved_output"
    }
    if not forward_primals or not forward_outputs:
        raise ReplayError(f"{label} forward role domains are empty")
    backward_tangents = 0
    for node in capture["backward"]["nodes"]:
        if node["op"] != "placeholder" or node["role"] is None:
            continue
        kind, index = _role_descriptor(node["role"], f"{label}.backward.{node['id']}.role")
        if kind == "primal":
            domain = forward_primals
        elif kind == "tangent":
            domain = forward_outputs
        elif kind == "saved_output":
            domain = forward_saved_outputs
        else:
            raise ReplayError(f"{label}.backward.{node['id']} role changed")
        if index not in domain:
            raise ReplayError(f"{label}.backward.{node['id']} role is outside the forward domain")
        if kind == "tangent":
            backward_tangents += 1
    if not backward_tangents:
        raise ReplayError(f"{label} backward role domains have no tangent placeholders")
    gradient_outputs: list[int] = []
    for role in capture["backward"]["output_roles"]:
        _, index = _role_descriptor(role, f"{label}.backward.output_role")
        if index not in forward_primals:
            raise ReplayError(f"{label} gradient role is outside the forward primal domain")
        gradient_outputs.append(index)
    expected_gradients = sorted(required_gradients)
    if gradient_outputs != expected_gradients:
        raise ReplayError(
            f"{label} gradient outputs {gradient_outputs} do not match "
            f"differentiable inputs {expected_gradients}"
        )


def _capture_diagnostics(raw: Any, label: str) -> tuple[set[int], set[int]]:
    diagnostics = _closed(raw, {"callbacks", "aot_autograd", "inputs", "grad_output"}, f"{label}.diagnostics")
    callbacks = _closed(diagnostics["callbacks"], {"forward", "backward"}, f"{label}.callbacks")
    for kind, value in callbacks.items():
        if _integer(value, f"{label}.callbacks.{kind}", minimum=1) != 1:
            raise ReplayError(f"{label}.callbacks.{kind} must equal one")
    counters = diagnostics["aot_autograd"]
    if not isinstance(counters, dict) or not counters or not all(isinstance(key, str) for key in counters):
        raise ReplayError(f"{label}.aot_autograd is malformed")
    for key, value in counters.items():
        _integer(value, f"{label}.aot_autograd.{key}", minimum=0)
    inputs = diagnostics["inputs"]
    if not isinstance(inputs, list) or not inputs or len(inputs) > 64:
        raise ReplayError(f"{label} runtime input metadata changed")
    required_gradients: set[int] = set()
    for index, item in enumerate(inputs):
        _runtime_tensor(item, f"{label}.diagnostics.inputs[{index}]")
        if item["requires_grad"]:
            required_gradients.add(index)
    if not required_gradients:
        raise ReplayError(f"{label} diagnostics contain no differentiable inputs")
    _runtime_tensor(diagnostics["grad_output"], f"{label}.diagnostics.grad_output")
    return set(range(len(inputs))), required_gradients


def _kernel(raw: Any) -> None:
    kernel = _closed(
        raw,
        {"id", "module", "qualname", "source_sha256", "source", "argument_names", "reads", "writes", "source_reads", "source_writes", "runtime"},
        "kernel",
    )
    if not all(isinstance(kernel[key], str) and kernel[key] for key in ("id", "module", "qualname", "source_sha256", "source")):
        raise ReplayError("kernel scalar identity fields are malformed")
    if not _is_digest(kernel["id"]) or not _is_digest(kernel["source_sha256"]):
        raise ReplayError("kernel digest identity fields are malformed")
    if len(kernel["source"].encode("utf-8")) > MAX_STRING_BYTES:
        raise ReplayError("kernel source exceeds the size limit")
    if _sha_bytes(kernel["source"].encode("utf-8")) != kernel["source_sha256"]:
        raise ReplayError("kernel source hash mismatch")
    for key in ("argument_names", "reads", "writes", "source_reads", "source_writes"):
        if not isinstance(kernel[key], list) or len(set(kernel[key])) != len(kernel[key]) or not all(isinstance(item, str) and item for item in kernel[key]):
            raise ReplayError(f"kernel.{key} must be a string list")
    if not kernel["writes"]:
        raise ReplayError("kernel has no source-bound writes")
    _kernel_runtime_record(kernel["runtime"])
    identity = {
        "module": kernel["module"],
        "qualname": kernel["qualname"],
        "source_sha256": kernel["source_sha256"],
        "arguments": kernel["argument_names"],
        "reads": kernel["reads"],
        "writes": kernel["writes"],
        "runtime": kernel["runtime"],
    }
    if _sha(identity) != kernel["id"]:
        raise ReplayError("kernel identity digest mismatch")
    if kernel["reads"] != kernel["source_reads"] or kernel["writes"] != kernel["source_writes"]:
        raise ReplayError("producer TTIR/source effect summaries disagree")
    arguments, reads, writes = _source_effects(kernel["source"], set(kernel["reads"]) | set(kernel["writes"]))
    if arguments != tuple(kernel["argument_names"]) or reads != tuple(kernel["reads"]) or writes != tuple(kernel["writes"]):
        raise ReplayError("kernel source signature/effects changed")


def _validate_hops(capture: dict[str, Any], label: str) -> None:
    kernels = _kernel_map(capture)
    for graph_name in ("forward", "backward"):
        graph = capture[graph_name]
        nodes = {node["id"]: node for node in graph["nodes"]}
        hop_writes: dict[str, dict[str, dict[str, Any]]] = {}
        selections: set[tuple[str, str]] = set()
        for node in graph["nodes"]:
            node_label = f"{label}.{graph_name}.{node['id']}"
            if node["target"] == "triton.functional":
                normalized = _closed(
                    node["kwargs"],
                    {"kernel", "bindings", "writes", "constants", "grid"},
                    f"{node_label}.kwargs",
                )
                _validate_grid(normalized["grid"], node_label)
                kernel = kernels.get(normalized["kernel"])
                if kernel is None:
                    raise ReplayError(f"{node_label} references an unknown kernel")
                bindings = normalized["bindings"]
                constants = normalized["constants"]
                valid_bindings = (
                    isinstance(bindings, dict)
                    and bool(bindings)
                    and all(isinstance(role, str) and role for role in bindings)
                )
                valid_constants = isinstance(constants, dict) and all(
                    isinstance(role, str) and role for role in constants
                )
                if not valid_bindings or not valid_constants:
                    raise ReplayError(f"{node_label} bindings/constants are malformed")
                if _contains_encoded_ref(constants):
                    raise ReplayError(f"{node_label}.constants contain encoded tensor refs")
                source_arguments, source_reads, source_writes = _source_effects(
                    kernel["source"],
                    set(kernel["reads"]) | set(kernel["writes"]),
                )
                required_roles = set(source_reads) | set(source_writes)
                invalid_signature = (
                    source_arguments != tuple(kernel["argument_names"])
                    or source_reads != tuple(kernel["reads"])
                    or source_writes != tuple(kernel["writes"])
                    or set(bindings) != required_roles
                    or bool(set(bindings) & set(constants))
                    or set(bindings) | set(constants) != set(source_arguments)
                )
                if invalid_signature:
                    raise ReplayError(f"{node_label} is not total over the source-bound signature/effects")
                if normalized["writes"] != sorted(source_writes):
                    raise ReplayError(f"{node_label} writes differ from source effects")
                bound_nodes = []
                for role, value in bindings.items():
                    reference = _reference(value)
                    bound = nodes.get(reference)
                    if bound is None or bound["tensor"] is None:
                        raise ReplayError(f"{node_label}.{role} is not bound to a prior tensor")
                    bound_nodes.append((role, reference, bound))
                write_nodes = [item for item in bound_nodes if item[0] in source_writes]
                write_refs = [item[1] for item in write_nodes]
                write_groups = [item[2]["tensor"]["alias_group"] for item in write_nodes]
                known_groups = [group for group in write_groups if group is not None]
                if len(write_refs) != len(set(write_refs)) or len(known_groups) != len(set(known_groups)):
                    raise ReplayError(f"{node_label} has aliased write destinations")
                hop_writes[node["id"]] = {
                    role: bound["tensor"]
                    for role, _, bound in write_nodes
                }
            elif node["target"] == "operator.getitem":
                args = _tuple(node["args"])
                if len(args) != 2 or node["kwargs"] != {}:
                    raise ReplayError(f"{node_label} is outside the exact getitem grammar")
                hop = _reference(args[0])
                key = args[1]
                expected = hop_writes.get(hop, {}).get(key) if isinstance(key, str) else None
                if expected is None:
                    raise ReplayError(f"{node_label} does not select a declared HOP write")
                selection = (hop, key)
                if selection in selections:
                    raise ReplayError(f"{node_label} duplicates a HOP output selection")
                selections.add(selection)
                if (
                    type(node["tensor"]) is not dict
                    or set(node["tensor"]) != set(expected)
                    or any(
                        node["tensor"][field] != expected[field]
                        for field in _HOP_WRITE_RESULT_FIELDS
                    )
                ):
                    raise ReplayError(f"{node_label} getitem result metadata changed")


def _validate_grid(raw: Any, label: str) -> None:
    if type(raw) is not list or len(raw) != 1:
        raise ReplayError(f"{label}.grid must contain one encoded launch tuple")
    encoded = raw[0]
    if not isinstance(encoded, dict) or set(encoded) != {"tuple"}:
        raise ReplayError(f"{label}.grid must contain an encoded launch tuple")
    dimensions = encoded["tuple"]
    if type(dimensions) is not list or len(dimensions) != 3:
        raise ReplayError(f"{label}.grid launch tuple must have three dimensions")
    product = 1
    for index, dimension in enumerate(dimensions):
        product *= _integer(
            dimension,
            f"{label}.grid[{index}]",
            minimum=1,
            maximum=(1 << 63) - 1,
        )
        if product > (1 << 63) - 1:
            raise ReplayError(f"{label}.grid product exceeds signed-64")


def _execute(capture: dict[str, Any]) -> dict[str, Any]:
    graph = capture["backward"]
    kernels = _kernel_map(capture)
    nodes = graph["nodes"]
    nodes_by_id = {node["id"]: node for node in nodes}
    states: dict[str, _State] = {}
    hop_outputs: dict[str, dict[str, _State]] = {}
    borrowed: list[dict[str, Any]] = []
    borrowed_parts: dict[tuple[str, str], str] = {}
    tangent_roles: set[str] = set()
    capture_digest = _sha(capture)
    hop_ordinal = 0
    storage_names = _canonical_storage_names(nodes)
    _validate_storage_identities(capture["forward"]["nodes"])
    _validate_storage_identities(nodes)
    for index, node in enumerate(nodes):
        node_id = node["id"]
        target = node["target"]
        if node["op"] == "placeholder":
            tangent = _tangent(node["role"])
            if tangent is not None:
                if tangent in tangent_roles:
                    raise ReplayError(f"duplicate tangent output role {tangent}")
                tangent_roles.add(tangent)
            role = node["role"] or {"kind": "saved", "slot": node_id}
            states[node_id] = _State(
                buffer=storage_names[node["tensor"]["alias_group"]],
                version=0,
                parts=((tangent,) if tangent else (f"input:{node_id}:{_sha(role)}",)),
                tangent=tangent,
                producer=node_id,
            )
        elif target in _ALIASES:
            if node["kwargs"] != {}:
                raise ReplayError(f"alias node {node_id} has unsupported keyword arguments")
            prior = _single(node["args"], states)
            states[node_id] = _State(prior.buffer, prior.version, prior.parts, prior.tangent, node_id)
        elif target in _ALLOCATIONS:
            _validate_allocation_grammar(node, nodes, index, f"backward.{node_id}")
            part = f"allocation:{_sha(_allocation_signature(node, states, capture_digest))}"
            states[node_id] = _State(storage_names[node["tensor"]["alias_group"]], 0, (part,), None, node_id)
        elif target == "aten.clone":
            if node["kwargs"] != {}:
                raise ReplayError(f"clone node {node_id} has unsupported keyword arguments")
            prior = _single(node["args"], states)
            states[node_id] = _State(storage_names[node["tensor"]["alias_group"]], 0, prior.parts, None, node_id)
        elif target == "triton.functional":
            normalized = _closed(node["kwargs"], {"kernel", "bindings", "writes", "constants", "grid"}, "HOP")
            kernel = kernels.get(normalized["kernel"])
            if kernel is None:
                raise ReplayError(f"HOP {node_id} references unknown kernel")
            raw_bindings = normalized["bindings"]
            if not isinstance(raw_bindings, dict) or not isinstance(normalized["writes"], list):
                raise ReplayError(f"HOP {node_id} bindings/writes are malformed")
            required_roles = set(kernel["reads"]) | set(kernel["writes"])
            if set(raw_bindings) != required_roles:
                raise ReplayError(f"HOP {node_id} bindings differ from source roles")
            bindings = {role: _state(value, states) for role, value in raw_bindings.items()}
            source_arguments, source_reads, source_writes = _source_effects(kernel["source"], required_roles)
            if source_arguments != tuple(kernel["argument_names"]):
                raise ReplayError(f"HOP {node_id} source signature changed")
            if source_reads != tuple(kernel["reads"]) or source_writes != tuple(kernel["writes"]):
                raise ReplayError(f"HOP {node_id} source effects do not replay")
            if tuple(sorted(normalized["writes"])) != source_writes:
                raise ReplayError(f"HOP {node_id} write roles changed")
            write_buffers = [bindings[role].buffer for role in source_writes]
            if len(set(write_buffers)) != len(write_buffers):
                raise ReplayError(f"HOP {node_id} has aliased write destinations")
            outputs: dict[str, _State] = {}
            for write_role in source_writes:
                before = bindings[write_role]
                write_layout = _binding_layout(nodes_by_id, raw_bindings[write_role])
                signature = {
                    "kernel": kernel["id"],
                    # Dense node IDs are capture-local: a paired control may add a
                    # fresh destination before the same logical HOP.  The ordinal
                    # still keeps distinct HOP occurrences separate while allowing
                    # exact candidate/control contribution matching.
                    "hop_ordinal": hop_ordinal,
                    "write_role": write_role,
                    "write_layout": write_layout,
                    "reads": {
                        role: {
                            "buffer": bindings[role].buffer,
                            "version": bindings[role].version,
                            "parts": list(bindings[role].parts),
                            "layout": _binding_layout(nodes_by_id, raw_bindings[role]),
                        }
                        for role in source_reads
                    },
                    "constants": normalized["constants"],
                    "grid": normalized["grid"],
                }
                part = f"hop:{_sha(signature)}"
                outputs[write_role] = _State(f"functional:{node_id}:{write_role}", before.version + 1, (part,), None, node_id)
                if before.tangent is not None:
                    read_roles = [
                        role
                        for role in source_reads
                        if bindings[role].buffer == before.buffer and bindings[role].version == before.version
                    ] or ["<none>"]
                    for read_role in read_roles:
                        borrowed_parts[(node_id, write_role)] = part
                        borrowed.append(
                            {
                                "tangent": before.tangent,
                                "hop_node": node_id,
                                "read_role": read_role,
                                "write_role": write_role,
                                "kernel_id": kernel["id"],
                                "before_version": before.version,
                                "after_version": before.version + 1,
                            }
                        )
            hop_outputs[node_id] = outputs
            hop_ordinal += 1
        elif target == "operator.getitem":
            args = _tuple(node["args"])
            if len(args) != 2 or node["kwargs"] != {}:
                raise ReplayError(f"getitem {node_id} is outside the exact two-argument grammar")
            hop = _reference(args[0])
            key = args[1]
            if hop not in hop_outputs or not isinstance(key, str) or key not in hop_outputs[hop]:
                raise ReplayError(f"getitem {node_id} has invalid HOP/key")
            source = nodes_by_id[hop]
            binding = source["kwargs"]["bindings"].get(key)
            bound = None if binding is None else nodes_by_id.get(_reference(binding))
            if (
                bound is None
                or bound["tensor"] is None
                or node["tensor"] is None
                or any(
                    node["tensor"][field] != bound["tensor"][field]
                    for field in _HOP_WRITE_RESULT_FIELDS
                )
            ):
                raise ReplayError(f"getitem {node_id} result metadata changed")
            prior = hop_outputs[hop][key]
            states[node_id] = _State(
                storage_names[node["tensor"]["alias_group"]],
                prior.version,
                prior.parts,
                prior.tangent,
                node_id,
            )
        elif target in _PURE:
            args = _tuple(node["args"])
            if len(args) != 2 or node["kwargs"] != {}:
                raise ReplayError(f"pure node {node_id} is outside the exact two-argument grammar")
            dependencies = _states(node["args"], states)
            if not dependencies:
                raise ReplayError(f"pure node {node_id} has no tensor input")
            signature = {
                "target": target,
                "parts": [list(state.parts) for state in dependencies],
                "literals": _without_refs(node["args"]),
                "kwargs": node["kwargs"],
            }
            states[node_id] = _State(
                storage_names[node["tensor"]["alias_group"]],
                0,
                (f"value:{_sha(signature)}",),
                None,
                node_id,
            )
        elif target == "aten.add":
            args = _tuple(node["args"])
            if len(args) != 2 or node["kwargs"] != {}:
                raise ReplayError(f"add node {node_id} is outside the exact two-operand grammar")
            left, right = _state(args[0], states), _state(args[1], states)
            if node["gradient_accumulation"]:
                if len(left.parts) + len(right.parts) > MAX_DERIVED_PARTS:
                    raise ReplayError(
                        f"add node {node_id} exceeds the derived-contribution work limit"
                    )
                parts = left.parts + right.parts
            else:
                parts = (f"value:{_sha({'target': target, 'parts': [left.parts, right.parts], 'args': _without_refs(node['args']), 'kwargs': node['kwargs']})}",)
            states[node_id] = _State(
                storage_names[node["tensor"]["alias_group"]],
                0,
                parts,
                None,
                node_id,
            )
        elif node["op"] == "output":
            pass
        else:
            raise ReplayError(f"unsupported replay node {node_id}: {node['op']}/{target}")
    output_refs = _output_refs(nodes[-1]["args"])
    roles = graph["output_roles"]
    if len(output_refs) != len(roles):
        raise ReplayError("backward output roles/values differ")
    gradients: dict[int, _State] = {}
    for value, role in zip(output_refs, roles):
        kind, index = _role_descriptor(role, "backward gradient output descriptor")
        if kind != "gradient" or index in gradients:
            raise ReplayError("duplicate or malformed gradient output slot")
        gradients[index] = _state(value, states)
    return {"gradients": gradients, "borrowed": borrowed, "borrowed_parts": borrowed_parts}


def _analysis(candidate, control, candidate_result, control_result):
    violation = bool(candidate_result["borrowed"])
    if control is None:
        return {
            "outcome": "VIOLATION" if violation else "SAFE_IN_FRAGMENT",
            "reasons": ["BORROWED_COTANGENT_WRITE"] if violation else ["FRESH_DESTINATION"],
            "borrowed_writes": candidate_result["borrowed"],
            "collapse": None,
            "unknown_detail": None,
            "candidate_digest": _sha(candidate),
            "control_digest": None,
        }
    if control_result is None:
        raise ReplayError("candidate/control gradient output slots differ")
    if control_result["borrowed"]:
        raise ReplayError("TWIN_RELATION: control graph contains borrowed writes")
    if set(candidate_result["gradients"]) != set(control_result["gradients"]):
        raise ReplayError("candidate/control gradient output slots differ")
    collapses = []
    borrowed_parts = {
        (candidate_result["borrowed_parts"][(item["hop_node"], item["write_role"])], item["tangent"])
        for item in candidate_result["borrowed"]
    }
    for slot in sorted(candidate_result["gradients"]):
        actual = candidate_result["gradients"][slot]
        expected = control_result["gradients"][slot]
        if Counter(actual.parts) == Counter(expected.parts):
            continue
        for lane, capture, state in (
            ("candidate", candidate, actual),
            ("control", control, expected),
        ):
            producer = next(
                (
                    node
                    for node in capture["backward"]["nodes"]
                    if node["id"] == state.producer
                ),
                None,
            )
            if (
                producer is None
                or producer["target"] != "aten.add"
                or producer["gradient_accumulation"] is not True
            ):
                raise ReplayError(
                    f"TWIN_RELATION: {lane} gradient slot {slot} is not produced "
                    "by a marked gradient-accumulation add"
                )
        matched = None
        for hop_part, borrowed_tangent in sorted(borrowed_parts):
            substituted = Counter(expected.parts)
            if substituted[borrowed_tangent] == 0:
                continue
            substituted[borrowed_tangent] -= 1
            if not substituted[borrowed_tangent]:
                del substituted[borrowed_tangent]
            substituted[hop_part] += 1
            if Counter(actual.parts) == substituted:
                if Counter(expected.parts)[borrowed_tangent] != 1:
                    raise ReplayError(
                        f"TWIN_RELATION: control gradient slot {slot} repeats {borrowed_tangent}"
                    )
                matched = {
                    "gradient_of": slot,
                    "accumulation_node": actual.producer,
                    "expected_parts": list(expected.parts),
                    "actual_parts": list(actual.parts),
                    "replaced_part": borrowed_tangent,
                    "replacement_part": hop_part,
                }
                break
        if matched is None:
            raise ReplayError(f"gradient slot {slot} is not an exact candidate/control contribution substitution")
        collapses.append(matched)
    if len(collapses) > 1:
        raise ReplayError("multiple collapsed gradient slots are outside the profile")
    if _backward_twin_skeleton(candidate) != _backward_twin_skeleton(control):
        raise ReplayError("TWIN_RELATION: normalized backward operator skeletons differ")
    collapse = collapses[0] if collapses else None
    reasons = ["BORROWED_COTANGENT_WRITE"] if violation else ["FRESH_DESTINATION"]
    if collapse:
        reasons.append("CONTRIBUTION_COLLAPSE")
    return {
        "outcome": "VIOLATION" if violation else "SAFE_IN_FRAGMENT",
        "reasons": reasons,
        "borrowed_writes": candidate_result["borrowed"],
        "collapse": collapse,
        "unknown_detail": None,
        "candidate_digest": _sha(candidate),
        "control_digest": _sha(control),
    }


def _canonical_storage_names(nodes: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for node in nodes:
        tensor = node.get("tensor")
        if not isinstance(tensor, dict) or tensor.get("alias_group") is None:
            continue
        names.setdefault(tensor["alias_group"], f"storage:a{len(names)}")
    return names


def _tangent(role):
    if role is None:
        return None
    kind, index = _role_descriptor(role, "backward placeholder role")
    if kind != "tangent":
        return None
    return f"tangent:output:{index}"
