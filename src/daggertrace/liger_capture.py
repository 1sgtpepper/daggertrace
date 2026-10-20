"""Liger capture specialization contract.

The target facade supplies the fixed shape so this module owns only the graph
contract, not target routing or runtime evidence.
"""

from __future__ import annotations

from typing import Any

from .errors import InvalidArtifact, TargetError, UnsupportedCapture
from .model import Capture
from .ownership_common import _graph_output_refs, _ref_id

def validate_liger_capture_specialization(
    candidate: Capture,
    control: Capture,
    *,
    shape: tuple[int, int],
    expected_partial_rows: int | None = None,
) -> None:
    """Bind the paired capture to the fixed row-mode Liger demonstration."""

    try:
        _validate_liger_capture_specialization_exact(
            candidate, control, shape, expected_partial_rows
        )
    except TargetError:
        raise
    except (
        InvalidArtifact,
        UnsupportedCapture,
        AttributeError,
        IndexError,
        KeyError,
        StopIteration,
        TypeError,
        ValueError,
    ) as exc:
        raise TargetError(f"Liger capture specialization is malformed: {exc}") from exc


def _validate_liger_capture_specialization_exact(
    candidate: Capture,
    control: Capture,
    shape: tuple[int, int],
    expected_partial_rows: int | None,
) -> None:
    rows, columns = shape
    weight_shape = (columns,)
    rstd_shape = (rows,)
    layouts = {
        shape: (columns, 1),
        weight_shape: (1,),
        rstd_shape: (1,),
    }
    expected_config = {
        "kwargs": {},
        "num_warps": 4,
        "num_ctas": 1,
        "num_stages": 3,
        "maxnreg": None,
        "pre_hook": None,
        "ir_override": None,
    }
    forward_constants = {
        "BLOCK_SIZE": columns,
        "RSTD_row_stride": 1,
        "W_row_stride": 1,
        "X_row_stride": columns,
        "Y_row_stride": columns,
        "casting_mode": 1,
        "elementwise_affine": True,
        "eps": 1e-6,
        "n_cols": columns,
        "offset": 1.0,
    }
    backward_constants = {
        "BLOCK_SIZE": columns,
        "RSTD_row_stride": 1,
        "W_row_stride": 1,
        "X_dtype": {"torch": "fp32"},
        "X_row_stride": columns,
        "casting_mode": 1,
        "dW_row_stride": columns,
        "dX_row_stride": columns,
        "dY_row_stride": columns,
        "elementwise_affine": True,
        "n_cols": columns,
        "n_rows": rows,
        "offset": 1.0,
    }
    # The predicates below bind every node in these graphs; counts close them
    # against unreachable additions.
    expected_node_counts = {"candidate": (11, 15), "control": (11, 14)}

    def ref(value: Any) -> str:
        return _ref_id(value, "Liger target node reference")

    def tensor(node: Any, shape: tuple[int, ...], device: str) -> bool:
        value = node.tensor if node is not None else None
        layout = layouts.get(shape, (shape[1], 1) if len(shape) == 2 else None)
        return value is not None and (
            tuple(value.shape),
            tuple(value.stride),
            value.storage_offset,
            value.dtype,
            value.device,
        ) == (shape, layout, 0, "float32", device)

    def runtime(value: Any, shape: tuple[int, ...], device: str, requires_grad: bool) -> bool:
        return isinstance(value, dict) and (
            tuple(value.get("shape", ())),
            tuple(value.get("stride", ())),
            value.get("storage_offset"),
            value.get("dtype"),
            value.get("device"),
            value.get("requires_grad"),
        ) == (shape, layouts[shape], 0, "float32", device, requires_grad)

    def one(nodes: Any, predicate: Any) -> Any:
        matches = [node for node in nodes if predicate(node)]
        return matches[0] if len(matches) == 1 else None

    def getitem(graph: Any, hop: Any, key: str) -> Any:
        return one(
            graph.nodes,
            lambda node: node.target == "operator.getitem"
            and isinstance(node.args, dict)
            and node.args.get("tuple") == [{"ref": hop.id}, key]
            and node.kwargs == {},
        )

    def view(
        node: Any,
        source: Any,
        shape: tuple[int, ...],
        dimensions: list[int],
        device: str,
    ) -> bool:
        return (
            node is not None
            and source is not None
            and node.target == "aten.view"
            and node.args == {"tuple": [{"ref": source.id}, dimensions]}
            and node.kwargs == {}
            and tensor(node, shape, device)
        )

    def allocation(
        node: Any,
        target: str,
        shape: tuple[int, ...],
        device: str,
        source: Any = None,
    ) -> bool:
        if source is None:
            args = {"tuple": [list(shape)]}
            kwargs = {
                "device": {"torch": device},
                "dtype": {"torch": "torch.float32"},
                "pin_memory": False,
            }
        else:
            args = {"tuple": [{"ref": source.id}]}
            kwargs = {"pin_memory": False}
        return (
            node is not None
            and node.target == target
            and node.args == args
            and node.kwargs == kwargs
            and tensor(node, shape, device)
        )

    def partial_weight_rows(node: Any, device: str) -> int | None:
        value = node.tensor if node is not None else None
        if value is None or len(value.shape) != 2:
            return None
        partial_rows, partial_columns = value.shape
        if (
            partial_rows <= 0
            or partial_columns != columns
            or tuple(value.stride) != (columns, 1)
            or value.storage_offset != 0
            or value.dtype != "float32"
            or value.device != device
        ):
            return None
        return partial_rows

    def output_refs(graph: Any, expected: tuple[Any, ...]) -> bool:
        values = _graph_output_refs(graph.nodes[-1].args)
        return len(values) == len(expected) and [ref(value) for value in values] == [
            node.id for node in expected
        ]

    def hop(graph: Any, kernels: dict[str, Any], qualname: str) -> Any:
        matches = [
            node
            for node in graph.nodes
            if node.target == "triton.functional"
            and isinstance(node.kwargs, dict)
            and (kernel := kernels.get(node.kwargs.get("kernel"))) is not None
            and kernel.qualname == qualname
        ]
        if len(matches) != 1 or sum(node.target == "triton.functional" for node in graph.nodes) != 1:
            raise TargetError("Liger HOP specialization changed")
        return matches[0]

    def check(capture: Capture, lane: str) -> str:
        diagnostics = capture.diagnostics
        inputs = diagnostics.get("inputs") if isinstance(diagnostics, dict) else None
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise TargetError(f"Liger {lane} input specialization changed")
        device = inputs[0].get("device") if isinstance(inputs[0], dict) else None
        suffix = device[5:] if isinstance(device, str) and device.startswith("cuda:") else ""
        if not isinstance(device, str) or (
            device != "cuda"
            and (not suffix.isascii() or not suffix.isdigit() or str(int(suffix)) != suffix)
        ):
            raise TargetError(f"Liger {lane} device specialization changed")
        if not runtime(inputs[0], shape, device, True) or not runtime(
            inputs[1], weight_shape, device, True
        ) or not runtime(diagnostics.get("grad_output"), shape, device, False):
            raise TargetError(f"Liger {lane} input metadata specialization changed")

        kernels = {kernel.id: kernel for kernel in capture.kernels}
        if any(kernel.runtime.get("config") != expected_config for kernel in capture.kernels):
            raise TargetError(f"Liger {lane} kernel schedule changed")

        forward, backward = capture.forward, capture.backward
        if (len(forward.nodes), len(backward.nodes)) != expected_node_counts[lane]:
            raise TargetError(f"Liger {lane} graph topology changed")
        if (
            forward.output_roles
            != (
                {"kind": "output", "index": 0},
                {"kind": "saved_output", "index": 0},
                {"kind": "saved_output", "index": 1},
                {"kind": "saved_output", "index": 2},
            )
            or backward.output_roles
            != (
                {"kind": "gradient", "primal": {"kind": "primal", "index": 0}},
                {"kind": "gradient", "primal": {"kind": "primal", "index": 1}},
            )
        ):
            raise TargetError(f"Liger {lane} output roles changed")

        forward_nodes = {node.id: node for node in forward.nodes}
        forward_x = one(
            forward.nodes,
            lambda node: node.role == {"kind": "primal", "index": 0}
            and node.target == "placeholder"
            and tensor(node, shape, device),
        )
        forward_weight = one(
            forward.nodes,
            lambda node: node.role == {"kind": "primal", "index": 1}
            and node.target == "placeholder"
            and tensor(node, weight_shape, device),
        )
        forward_hop = hop(forward, kernels, "_rms_norm_forward_kernel")
        forward_bindings = forward_hop.kwargs.get("bindings")
        if not isinstance(forward_bindings, dict) or set(forward_bindings) != {
            "X_ptr",
            "W_ptr",
            "Y_ptr",
            "RSTD_ptr",
        }:
            raise TargetError(f"Liger {lane} forward bindings changed")
        forward_x_view = forward_nodes.get(ref(forward_bindings["X_ptr"]))
        forward_y = forward_nodes.get(ref(forward_bindings["Y_ptr"]))
        forward_rstd = forward_nodes.get(ref(forward_bindings["RSTD_ptr"]))
        if (
            forward_x is None
            or forward_weight is None
            or ref(forward_bindings["W_ptr"]) != forward_weight.id
            or not view(forward_x_view, forward_x, shape, [-1, columns], device)
            or not allocation(forward_y, "aten.empty", shape, device)
            or not allocation(forward_rstd, "aten.empty", rstd_shape, device)
            or forward_hop.kwargs.get("constants") != forward_constants
            or forward_hop.kwargs.get("grid") != [{"tuple": [rows, 1, 1]}]
        ):
            raise TargetError(f"Liger {lane} forward specialization changed")
        forward_y_result = getitem(forward, forward_hop, "Y_ptr")
        forward_rstd_result = getitem(forward, forward_hop, "RSTD_ptr")
        forward_public_result = one(
            forward.nodes,
            lambda node: view(
                node,
                forward_y_result,
                shape,
                [rows, columns],
                device,
            ),
        )
        forward_public = one(
            forward.nodes,
            lambda node: node.target == "aten.add"
            and node.gradient_accumulation is False
            and tensor(node, shape, device)
            and forward_public_result is not None
            and node.args
            == {
                "tuple": [
                    {"ref": forward_x.id},
                    {"ref": forward_public_result.id},
                ]
            }
            and node.kwargs == {},
        )
        if (
            forward_y_result is None
            or forward_rstd_result is None
            or not tensor(forward_y_result, shape, device)
            or not tensor(forward_rstd_result, rstd_shape, device)
            or forward_public is None
            or not output_refs(
                forward,
                (forward_public, forward_weight, forward_x_view, forward_rstd_result),
            )
        ):
            raise TargetError(f"Liger {lane} forward output semantics changed")

        backward_nodes = {node.id: node for node in backward.nodes}
        backward_weight = one(
            backward.nodes,
            lambda node: node.role == {"kind": "primal", "index": 1}
            and node.target == "placeholder"
            and tensor(node, weight_shape, device),
        )
        backward_x = one(
            backward.nodes,
            lambda node: node.role is None
            and node.target == "placeholder"
            and tensor(node, shape, device),
        )
        backward_rstd = one(
            backward.nodes,
            lambda node: node.role is None
            and node.target == "placeholder"
            and tensor(node, rstd_shape, device),
        )
        backward_tangent = one(
            backward.nodes,
            lambda node: node.role
            == {"kind": "tangent", "output": {"kind": "output", "index": 0}}
            and node.target == "placeholder"
            and tensor(node, shape, device),
        )
        backward_hop = hop(backward, kernels, "_rms_norm_backward_kernel")
        backward_bindings = backward_hop.kwargs.get("bindings")
        if not isinstance(backward_bindings, dict) or set(backward_bindings) != {
            "X_ptr",
            "W_ptr",
            "RSTD_ptr",
            "dY_ptr",
            "dX_ptr",
            "dW_ptr",
        }:
            raise TargetError(f"Liger {lane} backward bindings changed")
        backward_dy = backward_nodes.get(ref(backward_bindings["dY_ptr"]))
        backward_dw = backward_nodes.get(ref(backward_bindings["dW_ptr"]))
        partial_rows = partial_weight_rows(backward_dw, device)
        if (
            backward_weight is None
            or backward_x is None
            or backward_rstd is None
            or backward_tangent is None
            or ref(backward_bindings["W_ptr"]) != backward_weight.id
            or ref(backward_bindings["X_ptr"]) != backward_x.id
            or ref(backward_bindings["RSTD_ptr"]) != backward_rstd.id
            or not view(backward_dy, backward_tangent, shape, [-1, columns], device)
            or partial_rows is None
            or not allocation(backward_dw, "aten.empty", (partial_rows, columns), device)
            or backward_hop.kwargs.get("constants")
            != {
                **backward_constants,
                "rows_per_program": (rows + partial_rows - 1) // partial_rows,
            }
            or backward_hop.kwargs.get("grid") != [{"tuple": [partial_rows, 1, 1]}]
        ):
            raise TargetError(f"Liger {lane} backward specialization changed")
        backward_dx_result = getitem(backward, backward_hop, "dX_ptr")
        backward_dw_result = getitem(backward, backward_hop, "dW_ptr")
        backward_weight_gradient = one(
            backward.nodes,
            lambda node: node.target == "aten.sum"
            and node.args == {"tuple": [{"ref": backward_dw_result.id}, [0]]}
            and node.kwargs == {}
            and tensor(node, weight_shape, device),
        ) if backward_dw_result is not None else None
        if (
            backward_dx_result is None
            or backward_dw_result is None
            or not tensor(backward_dx_result, shape, device)
            or not tensor(backward_dw_result, (partial_rows, columns), device)
            or backward_weight_gradient is None
        ):
            raise TargetError(f"Liger {lane} backward output semantics changed")

        if lane == "candidate":
            first_gradient = one(
                backward.nodes,
                lambda node: view(node, backward_dx_result, shape, [rows, columns], device),
            )
            second_gradient = one(
                backward.nodes,
                lambda node: view(node, first_gradient, shape, [-1, columns], device)
                if first_gradient is not None
                else False,
            )
            final_gradient = one(
                backward.nodes,
                lambda node: view(node, second_gradient, shape, [rows, columns], device)
                if second_gradient is not None
                else False,
            )
            gradient = one(
                backward.nodes,
                lambda node: node.target == "aten.add"
                and node.gradient_accumulation is True
                and tensor(node, shape, device)
                and first_gradient is not None
                and final_gradient is not None
                and node.args
                == {"tuple": [{"ref": first_gradient.id}, {"ref": final_gradient.id}]}
                and node.kwargs == {},
            )
            expected_destination = backward_dy
        else:
            destination = backward_nodes.get(ref(backward_bindings["dX_ptr"]))
            gradient_result = one(
                backward.nodes,
                lambda node: view(node, backward_dx_result, shape, [rows, columns], device),
            )
            gradient = one(
                backward.nodes,
                lambda node: node.target == "aten.add"
                and node.gradient_accumulation is True
                and tensor(node, shape, device)
                and gradient_result is not None
                and node.args
                == {"tuple": [{"ref": backward_tangent.id}, {"ref": gradient_result.id}]}
                and node.kwargs == {},
            )
            expected_destination = destination
            if not allocation(destination, "aten.zeros_like", shape, device, backward_dy):
                raise TargetError("Liger control gradient destination changed")
        if (
            ref(backward_bindings["dX_ptr"]) != expected_destination.id
            or gradient is None
            or not output_refs(backward, (gradient, backward_weight_gradient))
        ):
            raise TargetError(f"Liger {lane} gradient semantics changed")
        return device, partial_rows

    candidate_device, candidate_partial_rows = check(candidate, "candidate")
    control_device, control_partial_rows = check(control, "control")
    if candidate_device != control_device:
        raise TargetError("Liger candidate/control CUDA devices differ")
    if candidate_partial_rows != control_partial_rows:
        raise TargetError("Liger candidate/control partial row counts differ")
    if expected_partial_rows is not None and candidate_partial_rows != expected_partial_rows:
        raise TargetError("Liger capture partial rows differ from its runtime witness")
