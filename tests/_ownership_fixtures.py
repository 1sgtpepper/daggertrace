from __future__ import annotations

import copy

from daggertrace.canonical import sha256_bytes, sha256_json
from daggertrace.model import Capture
from daggertrace.validation_v2_fixtures import _micrograph

def _v2_functional_result_capture(target):
    candidate, _ = _micrograph(
        False,
        "materialized" if target in {"aten.mul", "aten.sum"} else "direct",
        False,
        1,
        False,
    )
    raw = candidate.to_dict()
    node = next(
        node
        for node in raw["backward"]["nodes"]
        if node["target"] == ("aten.mul" if target == "aten.sum" else target)
    )
    if target == "aten.sum":
        node.update(target="aten.sum", args={"tuple": [{"ref": "n0"}, []]})
    return raw, node

def _make_v2_alias(raw):
    alias = _node(raw, "aten.view")
    alias.update(
        target="aten.alias",
        args={"tuple": [copy.deepcopy(alias["args"]["tuple"][0])]}
    )

def _read_only_hop_raw():
    candidate, _ = _micrograph(False, "none", False, 1, False)
    raw = copy.deepcopy(candidate.to_dict())
    nodes = raw["backward"]["nodes"]
    hop = next(node for node in nodes if node["target"] == "triton.functional")
    hop["kwargs"]["writes"] = []
    getitem = next(node for node in nodes if node["target"] == "operator.getitem")
    nodes.remove(getitem)
    output = nodes[-1]
    output["id"] = f"n{len(nodes) - 1}"
    output["args"] = {"tuple": [{"tuple": [{"ref": "n1"}]}]}
    return raw

def _two_input_raw():
    candidate, _ = _micrograph(False, "none", False, 1, False)
    raw = copy.deepcopy(candidate.to_dict())
    forward = raw["forward"]
    primal = copy.deepcopy(forward["nodes"][0])
    primal.update(
        id="n1",
        role={"kind": "primal", "index": 1},
        tensor={**primal["tensor"], "alias_group": "primal-second"},
    )
    output = forward["nodes"][-1]
    output.update(
        id="n2",
        args={"tuple": [{"tuple": [{"ref": "n0"}, {"ref": "n1"}]}]},
    )
    forward["nodes"] = [forward["nodes"][0], primal, output]
    forward["output_roles"] = [
        {"kind": "output", "index": 0},
        {"kind": "saved_output", "index": 0},
    ]
    backward = raw["backward"]
    backward["nodes"][-1]["args"]["tuple"][0]["tuple"].append(
        copy.deepcopy(backward["nodes"][-1]["args"]["tuple"][0]["tuple"][0])
    )
    backward["output_roles"].append(
        {"kind": "gradient", "primal": {"kind": "primal", "index": 1}}
    )
    raw["diagnostics"]["inputs"].append(copy.deepcopy(raw["diagnostics"]["inputs"][0]))
    return raw

def _edited(capture, edit):
    raw = copy.deepcopy(capture.to_dict())
    edit(raw)
    return Capture.from_dict(raw)


def _with_explicit_saved_output(raw):
    forward_output = raw["forward"]["nodes"][-1]
    values = forward_output["args"]["tuple"][0]["tuple"]
    values.append(copy.deepcopy(values[0]))
    raw["forward"]["output_roles"].append(
        {"kind": "saved_output", "index": 1}
    )
    saved = next(
        node
        for node in raw["backward"]["nodes"]
        if node["target"] in {"aten.zeros_like", "aten.empty_like", "aten.empty", "aten.zeros"}
    )
    saved.update(
        op="placeholder",
        target="placeholder",
        args={"tuple": []},
        kwargs={},
        role={"kind": "saved_output", "index": 1},
    )
    return raw


def _sync_hop_getitems(raw):
    nodes = {node["id"]: node for node in raw["backward"]["nodes"]}
    for result in nodes.values():
        if result["target"] != "operator.getitem":
            continue
        hop, key = result["args"]["tuple"]
        source = nodes[hop["ref"]]
        binding = source["kwargs"]["bindings"].get(key)
        if binding is not None:
            alias_group = result["tensor"]["alias_group"]
            result["tensor"] = copy.deepcopy(nodes[binding["ref"]]["tensor"])
            result["tensor"]["alias_group"] = alias_group

def _redirect_hop_write_to_tangent(capture):
    def edit(raw):
        hop = _node(raw, "triton.functional")
        write_role = hop["kwargs"]["writes"][0]
        hop["kwargs"]["bindings"][write_role] = {"ref": _tangent_node(raw)["id"]}
        _sync_hop_getitems(raw)

    return _edited(capture, edit)

def _retarget_allocation(capture, target):
    raw = copy.deepcopy(capture.to_dict())
    allocation = _node(raw, "aten.zeros_like")
    if target == "aten.empty_like":
        args = {"tuple": [{"ref": "n0"}]}
    else:
        args = {"tuple": [[2, 2]]}
    allocation.update(target=target, args=args)
    if target == "aten.zeros":
        allocation["kwargs"] = {
            "dtype": {"torch": "float32"},
            "device": "accelerator",
        }
        allocation["tensor"].update(shape=[2, 2], stride=[2, 1])
    _sync_hop_getitems(raw)
    return Capture.from_dict(raw)

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

def _kernel_for_capture(capture):
    raw = capture.to_dict() if isinstance(capture, Capture) else capture
    hops = [
        node
        for node in raw["backward"]["nodes"]
        if node["target"] == "triton.functional"
    ]
    assert len(hops) == 1
    kernel_id = hops[0]["kwargs"]["kernel"]
    return next(kernel for kernel in raw["kernels"] if kernel["id"] == kernel_id)

def _constant_signature(capture):
    raw = copy.deepcopy(capture.to_dict())
    kernel = _kernel_for_capture(raw)
    signature = f"def {kernel['qualname']}({', '.join(kernel['argument_names'])}):"
    kernel["source"] = kernel["source"].replace(
        signature,
        signature[:-2] + ", BLOCK_SIZE: tl.constexpr):",
    )
    kernel["argument_names"].append("BLOCK_SIZE")
    kernel["source_sha256"] = sha256_bytes(kernel["source"].encode("utf-8"))
    kernel["id"] = sha256_json(
        {
            "module": kernel["module"],
            "qualname": kernel["qualname"],
            "source_sha256": kernel["source_sha256"],
            "arguments": kernel["argument_names"],
            "reads": kernel["reads"],
            "writes": kernel["writes"],
            "runtime": kernel["runtime"],
        }
    )
    for graph in (raw["forward"], raw["backward"]):
        for node in graph["nodes"]:
            if node["target"] == "triton.functional":
                node["kwargs"]["kernel"] = kernel["id"]
                node["kwargs"]["constants"] = {"BLOCK_SIZE": 128}
    return Capture.from_dict(raw)

def _rename_kernel_formals(capture, read_role, write_role):
    raw = copy.deepcopy(capture.to_dict())
    kernel = _kernel_for_capture(raw)
    old_kernel_id = kernel["id"]
    old_read_role = kernel["reads"][0]
    old_write_role = kernel["writes"][0]
    role_map = {old_read_role: read_role, old_write_role: write_role}
    for old_role, new_role in sorted(role_map.items(), key=lambda item: -len(item[0])):
        kernel["source"] = kernel["source"].replace(old_role, new_role)
    kernel["argument_names"] = [role_map.get(role, role) for role in kernel["argument_names"]]
    kernel["reads"] = [role_map.get(role, role) for role in kernel["reads"]]
    kernel["writes"] = [role_map.get(role, role) for role in kernel["writes"]]
    kernel["source_reads"] = [role_map.get(role, role) for role in kernel["source_reads"]]
    kernel["source_writes"] = [role_map.get(role, role) for role in kernel["source_writes"]]
    kernel["source_sha256"] = sha256_bytes(kernel["source"].encode("utf-8"))
    kernel["id"] = sha256_json(
        {
            "module": kernel["module"],
            "qualname": kernel["qualname"],
            "source_sha256": kernel["source_sha256"],
            "arguments": kernel["argument_names"],
            "reads": kernel["reads"],
            "writes": kernel["writes"],
            "runtime": kernel["runtime"],
        }
    )

    for graph in (raw["forward"], raw["backward"]):
        for node in graph["nodes"]:
            if node["target"] == "triton.functional":
                kwargs = node["kwargs"]
                if kwargs["kernel"] == old_kernel_id:
                    kwargs["kernel"] = kernel["id"]
                kwargs["bindings"] = {
                    role_map.get(role, role): value
                    for role, value in kwargs["bindings"].items()
                }
                kwargs["writes"] = [role_map.get(role, role) for role in kwargs["writes"]]
            elif node["target"] == "operator.getitem":
                values = node["args"]["tuple"]
                if values[1] == old_write_role:
                    values[1] = write_role
    return Capture.from_dict(raw)

def _redirect_accumulation_to_existing_empty(capture):
    raw = copy.deepcopy(capture.to_dict())
    allocation = _node(raw, "aten.empty")
    assert allocation["target"] == "aten.empty"
    assert allocation["args"] == {"tuple": [[1]]}
    allocation["args"] = {"tuple": [[2, 2]]}
    allocation["tensor"].update(shape=[2, 2], stride=[2, 1])
    accumulation = next(
        node for node in raw["backward"]["nodes"] if node["target"] == "aten.add"
    )
    accumulation["args"]["tuple"][1] = {"ref": allocation["id"]}
    return Capture.from_dict(raw)

def _forward_with_empty(capture, alias_group):
    def insert_empty(raw):
        nodes = raw["forward"]["nodes"]
        output = _output_node(raw, "forward")
        empty = copy.deepcopy(
            next(
                node
                for node in nodes
                if isinstance(node.get("role"), dict)
                and node["role"].get("kind") == "primal"
            )
        )
        empty.update(
            id=f"n{len(nodes) - 1}",
            op="call_function",
            target="aten.empty",
            args={"tuple": [[2, 2]]},
            kwargs={"dtype": {"torch": "float32"}, "device": "accelerator"},
            role=None,
        )
        empty["tensor"]["alias_group"] = alias_group
        output["id"] = f"n{len(nodes)}"
        nodes.insert(-1, empty)

    return _edited(capture, insert_empty)

def _paired_allocation_outputs():
    candidate, control = _micrograph(False, "none", False, 1, False)

    def reseal(source, shape, alias_group):
        raw = source.to_dict()
        allocation = _node(raw, "aten.zeros_like")
        allocation["target"] = "aten.empty"
        allocation["args"] = {"tuple": [list(shape)]}
        allocation["kwargs"] = {
            "dtype": {"torch": "float32"},
            "device": "accelerator",
        }
        allocation["tensor"].update(
            shape=list(shape),
            stride=[shape[1], 1],
            alias_group=alias_group,
        )
        _sync_hop_getitems(raw)
        _output_node(raw)["args"] = {
            "tuple": [{"tuple": [{"ref": allocation["id"]}]}]
        }
        return Capture.from_dict(raw)

    return (
        reseal(candidate, (2, 2), "candidate-empty"),
        reseal(control, (3, 3), "control-empty"),
    )

def _insert_connected_clone_before_accumulation(raw):
    nodes = raw["backward"]["nodes"]
    accumulation = next(node for node in nodes if node["target"] == "aten.add")
    insert_at = int(accumulation["id"][1:])
    source_ref = copy.deepcopy(accumulation["args"]["tuple"][1])
    source = next(node for node in nodes if node["id"] == source_ref["ref"])
    clone = copy.deepcopy(source)
    clone.update(
        id=f"n{insert_at}",
        target="aten.clone",
        args={"tuple": [source_ref]},
        kwargs={},
        role=None,
        gradient_accumulation=False,
    )
    clone["tensor"]["alias_group"] = "unexpected-clone-destination"

    def shift(value):
        if isinstance(value, dict):
            if set(value) == {"ref"}:
                reference = value["ref"]
                index = int(reference[1:])
                return {"ref": f"n{index + 1}" if index >= insert_at else reference}
            return {key: shift(child) for key, child in value.items()}
        if isinstance(value, list):
            return [shift(child) for child in value]
        return value

    for node in nodes:
        index = int(node["id"][1:])
        if index >= insert_at:
            node["id"] = f"n{index + 1}"
        node["args"] = shift(node["args"])
        node["kwargs"] = shift(node["kwargs"])
    nodes.insert(insert_at, clone)
    shifted_accumulation = next(node for node in nodes if node["target"] == "aten.add")
    shifted_accumulation["args"]["tuple"][1] = {"ref": clone["id"]}

def _node(raw, target):
    graphs = (raw["backward"],) if "backward" in raw else (raw,)
    return next(node for graph in graphs for node in graph["nodes"] if node["target"] == target)

def _tangent_node(raw):
    return next(
        node
        for node in raw["backward"]["nodes"]
        if isinstance(node.get("role"), dict) and node["role"].get("kind") == "tangent"
    )

def _tangent_nodes(raw):
    return [
        node
        for node in raw["backward"]["nodes"]
        if isinstance(node.get("role"), dict) and node["role"].get("kind") == "tangent"
    ]

def _output_node(raw, graph_kind="backward"):
    return next(node for node in raw[graph_kind]["nodes"] if node["op"] == "output")

def _two_tangent_graphs():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    candidate_raw = copy.deepcopy(candidate.to_dict())
    control_raw = copy.deepcopy(control.to_dict())
    _insert_second_tangent(candidate_raw)
    _insert_second_tangent(control_raw)
    return candidate_raw, control_raw

def _insert_second_tangent(raw):
    forward = raw["forward"]
    forward["output_roles"].append({"kind": "output", "index": 1})
    primal = next(
        node
        for node in forward["nodes"]
        if isinstance(node.get("role"), dict) and node["role"].get("kind") == "primal"
    )
    _output_node(raw, "forward")["args"]["tuple"][0]["tuple"].append(
        {"ref": primal["id"]}
    )

    nodes = raw["backward"]["nodes"]
    index = next(
        position
        for position, node in enumerate(nodes)
        if node["target"] == "aten.add" and node["gradient_accumulation"]
    )
    placeholder = copy.deepcopy(_tangent_node(raw))
    for node in nodes:
        _shift_refs(node["args"], index)
        _shift_refs(node["kwargs"], index)
        ordinal = int(node["id"][1:])
        if ordinal >= index:
            node["id"] = f"n{ordinal + 1}"
    placeholder["id"] = f"n{index}"
    placeholder["role"] = {"kind": "tangent", "output": {"kind": "output", "index": 1}}
    placeholder["tensor"]["alias_group"] = "second-tangent"
    nodes.insert(index, placeholder)

def _shift_refs(value, start):
    if isinstance(value, dict):
        if set(value) == {"ref"}:
            ordinal = int(value["ref"][1:])
            if ordinal >= start:
                value["ref"] = f"n{ordinal + 1}"
            return
        for child in value.values():
            _shift_refs(child, start)
    elif isinstance(value, list):
        for child in value:
            _shift_refs(child, start)

def _append_accumulation(raw, left, right):
    nodes = raw["backward"]["nodes"]
    output = _output_node(raw)
    prior_add = _node(raw, "aten.add")
    new_id = output["id"]
    output["id"] = f"n{int(new_id[1:]) + 1}"
    accumulation = copy.deepcopy(prior_add)
    accumulation["id"] = new_id
    accumulation["args"] = {"tuple": [{"ref": left}, {"ref": right}]}
    accumulation["tensor"]["alias_group"] = f"gradient-{new_id}"
    nodes.insert(-1, accumulation)
    output["args"] = {"tuple": [{"tuple": [{"ref": new_id}]}]}
