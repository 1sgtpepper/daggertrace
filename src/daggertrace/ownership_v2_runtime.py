from __future__ import annotations

from .canonical import sha256_json
from .errors import UnsupportedCapture
from .model import BorrowedWrite, Capture, Reason
from .ownership_common import (
    _ALIASES,
    _ALLOCATIONS,
    _GraphResult,
    _MAX_DERIVED_PARTS,
    _PURE,
    _State,
    _allocation_signature,
    _all_states,
    _canonical_storage_names,
    _contains_encoded_ref,
    _forward_output_domains,
    _forward_primal_slots,
    _graph_output_refs,
    _layout_for_binding,
    _one_state,
    _one_state_reference,
    _positional,
    _ref_id,
    _role_index,
    _state_from_ref,
    _tangent_identity,
    _without_refs,
)
from .ownership_v2_validate import _validate_v2_diagnostics


def _run_backward(capture: Capture) -> _GraphResult:
    graph = capture.backward
    kernels = {kernel.id: kernel for kernel in capture.kernels}
    tangent_outputs, saved_outputs = _forward_output_domains(capture)
    primal_inputs = _forward_primal_slots(capture)
    states: dict[str, _State] = {}
    hop_outputs: dict[str, dict[str, _State]] = {}
    borrowed_writes: list[BorrowedWrite] = []
    borrowed_parts: dict[tuple[str, str], str] = {}
    tangent_roles: set[str] = set()
    nodes = {node.id: node for node in graph.nodes}
    storage_names = _canonical_storage_names(graph.nodes)
    capture_digest = capture.digest()
    hop_ordinal = 0

    for node in graph.nodes:
        if node.op == "placeholder":
            if node.tensor is None:
                raise UnsupportedCapture(f"placeholder {node.id} lacks tensor metadata")
            if node.role is None:
                role = {"kind": "saved", "slot": node.id}
                tangent = None
            else:
                role = node.role
                tangent = _tangent_identity(
                    role,
                    tangent_outputs=tangent_outputs,
                    saved_outputs=saved_outputs,
                    primal_inputs=primal_inputs,
                    label=f"placeholder {node.id}",
                )
            if tangent is not None:
                if tangent in tangent_roles:
                    raise UnsupportedCapture(f"duplicate tangent output role {tangent}")
                tangent_roles.add(tangent)
                states[node.id] = _State(
                    buffer=storage_names[node.tensor.alias_group],
                    version=0,
                    parts=(tangent,),
                    borrowed_tangent=tangent,
                    producer=node.id,
                )
            else:
                states[node.id] = _State(
                    buffer=storage_names[node.tensor.alias_group],
                    version=0,
                    parts=(f"input:{node.id}:{sha256_json(role)}",),
                    borrowed_tangent=None,
                    producer=node.id,
                )
            continue

        if node.target in _ALIASES:
            _positional(node.args)
            if node.kwargs != {}:
                raise UnsupportedCapture(
                    f"alias node {node.id} is outside the exact positional grammar"
                )
            source_id = _one_state_reference(node, states)
            state = states[source_id]
            states[node.id] = _State(
                buffer=state.buffer,
                version=state.version,
                parts=state.parts,
                borrowed_tangent=state.borrowed_tangent,
                producer=node.id,
            )
            continue

        if node.target in _ALLOCATIONS:
            part = f"allocation:{sha256_json(_allocation_signature(node, states, capture_digest))}"
            states[node.id] = _State(
                buffer=storage_names[node.tensor.alias_group],
                version=0,
                parts=(part,),
                borrowed_tangent=None,
                producer=node.id,
            )
            continue

        if node.target == "aten.clone":
            source = _one_state(node, states)
            states[node.id] = _State(
                buffer=storage_names[node.tensor.alias_group],
                version=0,
                parts=source.parts,
                borrowed_tangent=None,
                producer=node.id,
            )
            continue

        if node.target == "triton.functional":
            if not isinstance(node.kwargs, dict) or set(node.kwargs) != {
                "kernel",
                "bindings",
                "writes",
                "constants",
                "grid",
            }:
                raise UnsupportedCapture(f"HOP {node.id} fields changed")
            kernel_id = node.kwargs.get("kernel") if isinstance(node.kwargs, dict) else None
            kernel = kernels.get(kernel_id)
            if kernel is None:
                raise UnsupportedCapture(f"HOP {node.id} references unknown kernel {kernel_id!r}")
            if kernel.reads != kernel.source_reads or kernel.writes != kernel.source_writes:
                raise UnsupportedCapture(f"kernel {kernel.id} source/TTIR effects disagree")
            bindings = node.kwargs.get("bindings")
            writes = node.kwargs.get("writes")
            if not isinstance(bindings, dict) or not isinstance(writes, list):
                raise UnsupportedCapture(f"HOP {node.id} lacks normalized bindings/writes")
            required_roles = set(kernel.reads) | set(kernel.writes)
            if set(bindings) != required_roles:
                raise UnsupportedCapture(
                    f"HOP {node.id} bindings differ from source roles: "
                    f"expected={sorted(required_roles)}, actual={sorted(bindings)}"
                )
            constants = node.kwargs["constants"]
            if not isinstance(constants, dict) or not all(
                isinstance(role, str) and role for role in constants
            ):
                raise UnsupportedCapture(f"HOP {node.id} constants are malformed")
            if _contains_encoded_ref(constants):
                raise UnsupportedCapture(f"HOP {node.id} constants contain encoded tensor refs")
            if set(bindings) & set(constants) or set(bindings) | set(constants) != set(
                kernel.argument_names
            ):
                raise UnsupportedCapture(f"HOP {node.id} is not total over the kernel signature")
            bound: dict[str, _State] = {}
            for role, value in bindings.items():
                bound[role] = _state_from_ref(value, states, f"HOP {node.id}.{role}")
            if tuple(sorted(writes)) != kernel.writes:
                raise UnsupportedCapture(f"HOP {node.id} writes disagree with source-bound kernel")
            write_buffers = [bound[role].buffer for role in kernel.writes]
            if len(set(write_buffers)) != len(write_buffers):
                raise UnsupportedCapture(
                    f"{Reason.REGION_ALIAS.value}: HOP {node.id} has aliased write destinations"
                )
            outputs: dict[str, _State] = {}
            for write_role in kernel.writes:
                before = bound[write_role]
                binding_ref = _ref_id(bindings[write_role], f"HOP {node.id}.{write_role}")
                binding_tensor = nodes[binding_ref].tensor
                if binding_tensor is None:
                    raise UnsupportedCapture(f"HOP {node.id}.{write_role} lacks tensor metadata")
                write_layout = binding_tensor.to_dict()
                write_layout.pop("alias_group")
                signature = {
                    "kernel": kernel.id,
                    "hop_ordinal": hop_ordinal,
                    "write_role": write_role,
                    "write_layout": write_layout,
                    "reads": {
                        role: {
                            "buffer": bound[role].buffer,
                            "version": bound[role].version,
                            "parts": list(bound[role].parts),
                            "layout": _layout_for_binding(
                                nodes, bindings[role], f"HOP {node.id}.{role}"
                            ),
                        }
                        for role in kernel.reads
                    },
                    "constants": constants,
                    "grid": node.kwargs.get("grid"),
                }
                part = f"hop:{sha256_json(signature)}"
                outputs[write_role] = _State(
                    buffer=f"functional:{node.id}:{write_role}",
                    version=before.version + 1,
                    parts=(part,),
                    borrowed_tangent=None,
                    producer=node.id,
                )
                if before.borrowed_tangent is None:
                    continue
                read_roles = tuple(
                    role
                    for role in kernel.reads
                    if role in bound
                    and bound[role].buffer == before.buffer
                    and bound[role].version == before.version
                )
                if not read_roles:
                    read_roles = ("<none>",)
                for read_role in read_roles:
                    borrowed_parts[(node.id, write_role)] = part
                    borrowed_writes.append(
                        BorrowedWrite(
                            tangent=before.borrowed_tangent,
                            hop_node=node.id,
                            read_role=read_role,
                            write_role=write_role,
                            kernel_id=kernel.id,
                            before_version=before.version,
                            after_version=before.version + 1,
                        )
                    )
            hop_outputs[node.id] = outputs
            hop_ordinal += 1
            continue

        if node.target == "operator.getitem":
            values = _positional(node.args)
            if len(values) != 2 or node.kwargs != {}:
                raise UnsupportedCapture(
                    f"getitem {node.id} is outside the exact two-argument grammar"
                )
            hop_ref = _ref_id(values[0], f"getitem {node.id} source")
            key = values[1]
            if not isinstance(key, str) or hop_ref not in hop_outputs or key not in hop_outputs[hop_ref]:
                raise UnsupportedCapture(f"getitem {node.id} has unknown HOP key {key!r}")
            value = hop_outputs[hop_ref][key]
            states[node.id] = _State(
                buffer=storage_names[node.tensor.alias_group],
                version=value.version,
                parts=value.parts,
                borrowed_tangent=value.borrowed_tangent,
                producer=node.id,
            )
            continue

        if node.target in _PURE:
            values = _positional(node.args)
            if len(values) != 2 or node.kwargs != {}:
                raise UnsupportedCapture(
                    f"pure node {node.id} is outside the exact two-argument grammar"
                )
            dependencies = _all_states(node.args, states)
            if not dependencies:
                raise UnsupportedCapture(f"pure node {node.id} has no tensor input")
            signature = {
                "target": node.target,
                "parts": [list(state.parts) for state in dependencies],
                "literals": _without_refs(node.args),
                "kwargs": node.kwargs,
            }
            part = f"value:{sha256_json(signature)}"
            states[node.id] = _State(
                buffer=storage_names[node.tensor.alias_group],
                version=0,
                parts=(part,),
                borrowed_tangent=None,
                producer=node.id,
            )
            continue

        if node.target == "aten.add":
            values = _positional(node.args)
            if len(values) != 2 or node.kwargs != {}:
                raise UnsupportedCapture(f"add {node.id} is outside the exact two-operand grammar")
            left = _state_from_ref(values[0], states, f"add {node.id}.left")
            right = _state_from_ref(values[1], states, f"add {node.id}.right")
            if node.gradient_accumulation:
                if len(left.parts) + len(right.parts) > _MAX_DERIVED_PARTS:
                    raise UnsupportedCapture(
                        f"add {node.id} exceeds the derived-contribution work limit"
                    )
                parts = left.parts + right.parts
            else:
                parts = (
                    f"value:{sha256_json({'target': node.target, 'parts': [left.parts, right.parts], 'args': _without_refs(node.args), 'kwargs': node.kwargs})}",
                )
            states[node.id] = _State(
                buffer=storage_names[node.tensor.alias_group],
                version=0,
                parts=parts,
                borrowed_tangent=None,
                producer=node.id,
            )
            continue

        if node.op == "output":
            continue
        raise UnsupportedCapture(f"unsupported backward node {node.id}: {node.op}/{node.target}")

    if not tangent_roles:
        raise UnsupportedCapture("backward graph has no tangent roles")
    if not graph.output_roles:
        raise UnsupportedCapture("backward graph has no gradient outputs")
    expected_gradients = _validate_v2_diagnostics(capture.diagnostics, primal_inputs)
    output_refs = _graph_output_refs(graph.output_node().args)
    if len(output_refs) != len(graph.output_roles):
        raise UnsupportedCapture("backward output roles do not align with graph outputs")
    gradients: dict[int, _State] = {}
    gradient_order: list[int] = []
    for reference, role in zip(output_refs, graph.output_roles):
        if not isinstance(role, dict) or set(role) != {"kind", "primal"}:
            raise UnsupportedCapture(f"unsupported backward output descriptor {role}")
        if role["kind"] != "gradient":
            raise UnsupportedCapture(f"unsupported backward output descriptor {role}")
        primal = role["primal"]
        index = _role_index(primal, "primal", "gradient primal descriptor")
        if index not in primal_inputs:
            raise UnsupportedCapture(f"gradient output slot {index} is outside the primal domain")
        if index in gradients:
            raise UnsupportedCapture(f"duplicate gradient output slot {index}")
        gradients[index] = _state_from_ref(reference, states, "backward output")
        gradient_order.append(index)
    if tuple(gradient_order) != tuple(sorted(expected_gradients)):
        raise UnsupportedCapture("backward gradient projection does not match differentiable inputs")
    return _GraphResult(
        gradients=gradients,
        borrowed_writes=tuple(borrowed_writes),
        borrowed_parts=borrowed_parts,
    )
