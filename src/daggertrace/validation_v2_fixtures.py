"""Construct and transform the closed V2 capture fixtures."""

from __future__ import annotations

import copy
from typing import Any

from .canonical import sha256_bytes, sha256_json
from .model import Capture, Graph, Kernel, Node, TensorMeta, ref


def _append_node(nodes: list[Node], **fields: Any) -> str:
    node_id = f"n{len(nodes)}"
    nodes.append(Node(id=node_id, **fields))
    return node_id


def _first_tangent_placeholder(nodes: list[Node]) -> str:
    return next(
        node.id
        for node in nodes
        if node.op == "placeholder"
        and isinstance(node.role, dict)
        and node.role.get("kind") == "tangent"
    )


def _micrograph(
    alias_destination: bool,
    contribution: str,
    reverse_operands: bool,
    view_depth: int,
    noise: bool,
    *,
    allocation: str = "zeros_like",
    hops: int = 1,
) -> tuple[Capture, Capture]:
    if allocation not in {"zeros_like", "empty"}:
        raise AssertionError(f"unknown micrograph allocation {allocation!r}")
    if not isinstance(hops, int) or hops not in {1, 2, 3}:
        raise AssertionError(f"unsupported micrograph HOP count {hops!r}")

    kernels = []
    for ordinal in range(hops):
        read_role = f"read_role_{ordinal}"
        write_role = f"write_role_{ordinal}"
        kernel_source = (
            "@triton.jit\n"
            f"def kernel_{ordinal}({read_role}, {write_role}):\n"
            f"    value = tl.load({read_role})\n"
            f"    tl.store({write_role}, value)\n"
        )
        source_sha = sha256_bytes(kernel_source.encode("utf-8"))
        kernel_runtime = {
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
            "jit_cache_key": sha256_bytes(f"fixture-jit-cache-{ordinal}".encode("utf-8")),
        }
        identity = {
            "module": "fixture.module",
            "qualname": f"kernel_{ordinal}",
            "source_sha256": source_sha,
            "arguments": [read_role, write_role],
            "reads": [read_role],
            "writes": [write_role],
            "runtime": kernel_runtime,
        }
        kernels.append(
            Kernel(
                id=sha256_json(identity),
                module="fixture.module",
                qualname=f"kernel_{ordinal}",
                source_sha256=source_sha,
                source=kernel_source,
                argument_names=(read_role, write_role),
                reads=(read_role,),
                writes=(write_role,),
                source_reads=(read_role,),
                source_writes=(write_role,),
                runtime=kernel_runtime,
            )
        )
    forward = _micro_forward()

    def backward(alias: bool) -> Graph:
        nodes: list[Node] = []
        tangent = _append_node(
            nodes,
            op="placeholder",
            target="placeholder",
            args={"tuple": []},
            kwargs={},
            role={"kind": "tangent", "output": {"kind": "output", "index": 0}},
            tensor=_meta(alias_group="tangent"),
        )
        for _ in range(view_depth):
            tangent = _append_node(
                nodes,
                op="call_function",
                target="aten.view",
                args={"tuple": [ref(tangent), [2, 2]]},
                kwargs={},
                tensor=_meta(alias_group="tangent"),
            )
        materialized = None
        if contribution == "materialized":
            materialized = _append_node(
                nodes,
                op="call_function",
                target="aten.mul",
                args={"tuple": [ref(_first_tangent_placeholder(nodes)), 0.37]},
                kwargs={},
                tensor=_meta(alias_group="materialized"),
            )
        if noise:
            _append_node(
                nodes,
                op="call_function",
                target="aten.empty",
                args={"tuple": [[1]]},
                kwargs={"dtype": {"torch": "float32"}, "device": "accelerator"},
                tensor=_meta((1,), (1,), "noise"),
            )

        results: list[str] = []
        for ordinal, kernel in enumerate(kernels):
            read_role = kernel.reads[0]
            write_role = kernel.writes[0]
            if ordinal == 0 and alias:
                destination = tangent
            elif allocation == "zeros_like":
                alias_group = "destination" if ordinal == 0 else f"destination-{ordinal}"
                destination = _append_node(
                    nodes,
                    op="call_function",
                    target="aten.zeros_like",
                    args={"tuple": [ref(tangent)]},
                    kwargs={},
                    tensor=_meta(alias_group=alias_group),
                )
            else:
                alias_group = "destination" if ordinal == 0 else f"destination-{ordinal}"
                destination = _append_node(
                    nodes,
                    op="call_function",
                    target="aten.empty",
                    args={"tuple": [[2, 2]]},
                    kwargs={"dtype": {"torch": "float32"}, "device": "accelerator"},
                    tensor=_meta(alias_group=alias_group),
                )
            hop_id = _append_node(
                nodes,
                op="call_function",
                target="triton.functional",
                args=[],
                kwargs={
                    "kernel": kernel.id,
                    "bindings": {read_role: ref(tangent), write_role: ref(destination)},
                    "writes": [write_role],
                    "constants": {},
                    "grid": [{"tuple": [1, 1, 1]}],
                },
            )
            results.append(
                _append_node(
                    nodes,
                    op="call_function",
                    target="operator.getitem",
                    args={"tuple": [ref(hop_id), write_role]},
                    kwargs={},
                    tensor=_meta(alias_group=f"functional-{ordinal}"),
                )
            )

        output_id = results[0]
        if contribution != "none":
            if contribution == "direct":
                old = results[0] if alias else _first_tangent_placeholder(nodes)
            else:
                old = materialized
            operands = [ref(old), ref(results[0])]
            if reverse_operands:
                operands.reverse()
            output_id = _append_node(
                nodes,
                op="call_function",
                target="aten.add",
                args={"tuple": operands},
                kwargs={},
                tensor=_meta(alias_group="gradient-0"),
                gradient_accumulation=True,
            )
            for add_index, result_id in enumerate(results[1:], start=1):
                output_id = _append_node(
                    nodes,
                    op="call_function",
                    target="aten.add",
                    args={"tuple": [ref(output_id), ref(result_id)]},
                    kwargs={},
                    tensor=_meta(alias_group=f"gradient-{add_index}"),
                    gradient_accumulation=True,
                )
        _append_node(
            nodes,
            op="output",
            target="output",
            args={"tuple": [{"tuple": [ref(output_id)]}]},
            kwargs={},
        )
        return Graph(
            kind="backward",
            nodes=tuple(nodes),
            output_roles=({"kind": "gradient", "primal": {"kind": "primal", "index": 0}},),
        )

    framework = {
        "python": "3.14.6",
        "torch": "2.12.1+cu130",
        "torch_git": "7269437d655783a26cba32aa88195b741ff496aa",
        "triton": "3.7.1",
        "cuda_runtime": "13.0",
    }
    capture_config = {"fullgraph": True, "dynamic": False, "stage": "post-dynamo-pre-inductor"}
    candidate = Capture(
        label="candidate",
        framework=framework,
        capture_config=capture_config,
        forward=forward,
        backward=backward(alias_destination),
        kernels=tuple(kernels),
        diagnostics=copy.deepcopy(_MICROGRAPH_DIAGNOSTICS),
    )
    control = Capture(
        label="control",
        framework=framework,
        capture_config=capture_config,
        forward=forward,
        backward=backward(False),
        kernels=tuple(kernels),
        diagnostics=copy.deepcopy(_MICROGRAPH_DIAGNOSTICS),
    )
    return candidate, control


def _micro_forward() -> Graph:
    nodes: list[Node] = []
    primal = _append_node(
        nodes,
        op="placeholder",
        target="placeholder",
        args={"tuple": []},
        kwargs={},
        role={"kind": "primal", "index": 0},
        tensor=_meta(alias_group="primal"),
    )
    _append_node(
        nodes,
        op="output",
        target="output",
        args={"tuple": [{"tuple": [ref(primal)]}]},
        kwargs={},
    )
    return Graph(
        kind="forward",
        nodes=tuple(nodes),
        output_roles=({"kind": "output", "index": 0},),
    )


def _meta(shape=(2, 2), stride=(2, 1), alias_group="value"):
    return TensorMeta(
        shape=shape,
        stride=stride,
        dtype="float32",
        device="accelerator",
        alias_group=alias_group,
    )


_MICROGRAPH_DIAGNOSTICS = {
    "callbacks": {"forward": 1, "backward": 1},
    "aot_autograd": {"ok": 1, "total": 1},
    "inputs": [
        {
            "shape": [2, 2],
            "stride": [2, 1],
            "storage_offset": 0,
            "dtype": "float32",
            "device": "cuda:0",
            "requires_grad": True,
        }
    ],
    "grad_output": {
        "shape": [2, 2],
        "stride": [2, 1],
        "storage_offset": 0,
        "dtype": "float32",
        "device": "cuda:0",
        "requires_grad": False,
    },
}


def _edit_capture(capture: Capture, edit) -> Capture:
    raw = copy.deepcopy(capture.to_dict())
    edit(raw)
    return Capture.from_dict(raw)


def _node_with_target(capture: dict[str, Any], target: str, graph_kind: str = "backward"):
    return next(node for node in capture[graph_kind]["nodes"] if node["target"] == target)


def _alternate_tensor_reference(capture: dict[str, Any], node: dict[str, Any]) -> dict[str, str]:
    nodes = capture["backward"]["nodes"]
    node_index = next(index for index, item in enumerate(nodes) if item["id"] == node["id"])
    bound = node.get("kwargs", {}).get("bindings", {})
    excluded = {
        value.get("ref")
        for value in bound.values()
        if isinstance(value, dict) and set(value) == {"ref"}
    }
    for candidate in nodes[:node_index]:
        if candidate["id"] in excluded or candidate.get("tensor") is None:
            continue
        return ref(candidate["id"])
    raise AssertionError(f"node {node['id']} has no alternate prior tensor reference")


def _relabel_capture(capture: Capture) -> Capture:
    raw = copy.deepcopy(capture.to_dict())
    raw["label"] = f"renamed-{raw['label']}"
    kernel_ids: dict[str, str] = {}
    for kernel in raw["kernels"]:
        old_id = kernel["id"]
        kernel["module"] = "renamed.module"
        kernel["qualname"] = f"renamed_{kernel['qualname']}"
        kernel["id"] = sha256_json(_kernel_identity(kernel))
        kernel_ids[old_id] = kernel["id"]
    for graph in (raw["forward"], raw["backward"]):
        for node in graph["nodes"]:
            if node["target"] != "triton.functional":
                continue
            old_id = node["kwargs"]["kernel"]
            node["kwargs"]["kernel"] = kernel_ids[old_id]
    return Capture.from_dict(raw)


def _alpha_rename_capture(capture: Capture) -> Capture:
    raw = copy.deepcopy(capture.to_dict())
    kernel_ids: dict[str, str] = {}
    role_maps: dict[str, dict[str, str]] = {}
    renamed_role_maps: dict[str, dict[str, str]] = {}
    for kernel in raw["kernels"]:
        old_id = kernel["id"]
        role_map = {role: f"formal_{role}" for role in kernel["argument_names"]}
        role_maps[old_id] = role_map
        for field in ("argument_names", "reads", "writes", "source_reads", "source_writes"):
            kernel[field] = [role_map.get(role, role) for role in kernel[field]]
        source = kernel["source"]
        for old_role, new_role in sorted(role_map.items(), key=lambda item: -len(item[0])):
            source = source.replace(old_role, new_role)
        kernel["source"] = source
        kernel["source_sha256"] = sha256_bytes(source.encode("utf-8"))
        kernel["id"] = sha256_json(_kernel_identity(kernel))
        kernel_ids[old_id] = kernel["id"]
        renamed_role_maps[kernel["id"]] = role_map
    for graph in (raw["forward"], raw["backward"]):
        for node in graph["nodes"]:
            if node["target"] != "triton.functional":
                continue
            old_id = node["kwargs"]["kernel"]
            role_map = role_maps[old_id]
            node["kwargs"]["kernel"] = kernel_ids[old_id]
            node["kwargs"]["bindings"] = {
                role_map.get(role, role): value
                for role, value in node["kwargs"]["bindings"].items()
            }
            node["kwargs"]["writes"] = [
                role_map.get(role, role) for role in node["kwargs"]["writes"]
            ]
        for node in graph["nodes"]:
            if node["target"] != "operator.getitem":
                continue
            hop_id = node["args"]["tuple"][0]["ref"]
            hop = next(item for item in graph["nodes"] if item["id"] == hop_id)
            role_map = renamed_role_maps[hop["kwargs"]["kernel"]]
            node["args"]["tuple"][1] = role_map.get(
                node["args"]["tuple"][1], node["args"]["tuple"][1]
            )
    return Capture.from_dict(raw)


def _unsupported_kernel_capture(capture: Capture, expression: str) -> Capture:
    raw = copy.deepcopy(capture.to_dict())
    kernel = _backward_kernel(raw)
    expression = expression.replace("read_role_0", kernel["reads"][0])
    expression = expression.replace("write_role_0", kernel["writes"][0])
    kernel["source"] = f"{kernel['source'].rstrip()}\n    {expression}\n"
    kernel["source_sha256"] = sha256_bytes(kernel["source"].encode("utf-8"))
    old_id = kernel["id"]
    kernel["id"] = sha256_json(_kernel_identity(kernel))
    for graph in (raw["forward"], raw["backward"]):
        for node in graph["nodes"]:
            if node["target"] == "triton.functional" and node["kwargs"]["kernel"] == old_id:
                node["kwargs"]["kernel"] = kernel["id"]
    kernels = tuple(
        Kernel(
            id=item["id"],
            module=item["module"],
            qualname=item["qualname"],
            source_sha256=item["source_sha256"],
            source=item["source"],
            argument_names=tuple(item["argument_names"]),
            reads=tuple(item["reads"]),
            writes=tuple(item["writes"]),
            source_reads=tuple(item["source_reads"]),
            source_writes=tuple(item["source_writes"]),
            runtime=item["runtime"],
        )
        for item in raw["kernels"]
    )
    return Capture(
        schema=raw["schema"],
        label=raw["label"],
        framework=raw["framework"],
        capture_config=raw["capture_config"],
        forward=Graph.from_dict(raw["forward"]),
        backward=Graph.from_dict(raw["backward"]),
        kernels=kernels,
        diagnostics=raw["diagnostics"],
    )


def _kernel_identity(kernel: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": kernel["module"],
        "qualname": kernel["qualname"],
        "source_sha256": kernel["source_sha256"],
        "arguments": kernel["argument_names"],
        "reads": kernel["reads"],
        "writes": kernel["writes"],
        "runtime": kernel["runtime"],
    }


def _hop(capture, graph_kind="backward"):
    hops = [node for node in capture[graph_kind]["nodes"] if node["target"] == "triton.functional"]
    if len(hops) != 1:
        raise AssertionError("capture helper requires one HOP; select a kernel explicitly")
    return next(iter(hops))


def _backward_kernel(capture):
    kernel_id = _hop(capture)["kwargs"]["kernel"]
    return next(kernel for kernel in capture["kernels"] if kernel["id"] == kernel_id)


def _fresh_backward_allocation(capture):
    hop = _hop(capture)
    write_role = hop["kwargs"]["writes"][0]
    destination = hop["kwargs"]["bindings"][write_role]
    destination_id = destination["ref"]
    return next(
        node
        for node in capture["backward"]["nodes"]
        if node["target"] in {"aten.zeros_like", "aten.empty", "aten.empty_like"}
        and node["id"] == destination_id
    )


def _getitem(capture, key=None):
    nodes = [node for node in capture["backward"]["nodes"] if node["target"] == "operator.getitem"]
    if key is None:
        if not nodes:
            raise AssertionError("capture helper requires one getitem")
        # The corruption mutation only needs one admitted output; replay still
        # validates every getitem in the captured graph.
        return nodes[0]
    return next(node for node in nodes if node["args"]["tuple"][1] == key)


def _tangent_node(capture):
    return next(
        node
        for node in capture["backward"]["nodes"]
        if isinstance(node.get("role"), dict) and node["role"].get("kind") == "tangent"
    )


def _gradient_add(capture):
    return next(
        node
        for node in capture["backward"]["nodes"]
        if node["target"] == "aten.add" and node["gradient_accumulation"]
    )
