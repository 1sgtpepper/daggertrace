"""ART capture and specialization contract.

The adapter facade owns routing and target identity; this module owns only the
live capture boundary and exact graph profile.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from . import capture as _capture_module
from .capture import capture_aot
from .errors import InvalidArtifact, TargetError, UnsupportedCapture
from .model import Capture
from .ownership_common import _graph_output_refs, _ref_id
from .source_files import _load_exact_module


def _target_module():
    """Load the facade only after its module initialization has completed."""

    from . import art_target

    return art_target


def _art_function(module: types.ModuleType) -> Callable[[Any], Any]:
    target = _target_module()

    def function(logits):
        return module.local_topk_stats(logits, k=target.ART_K)[1]

    return function


def _reference_art_function() -> Callable[[Any], Any]:
    target = _target_module()
    torch = target._require_module("torch", "PyTorch", "ART")

    def function(logits):
        row_max = logits.detach().amax(dim=-1)
        return torch.exp(logits - row_max.unsqueeze(-1)).sum(dim=-1)

    return function


def _capture_art(function, logits0, grad_output) -> tuple[Capture, list[dict[str, Any]]]:
    logits = logits0.clone().requires_grad_(True)
    with _art_capture_context() as observations:
        capture = capture_aot(
            function,
            (logits,),
            grad_output.clone(),
            label="openpipe-art-local-stats-k2",
        )
    return capture, observations


@contextmanager
def _art_capture_context() -> Iterator[list[dict[str, Any]]]:
    """Observe live HOP wrappers; generic capture owns the closed language."""
    target = _target_module()

    with target._ART_CAPTURE_LOCK:
        original_runtime = _capture_module._kernel_runtime
        observations: list[dict[str, Any]] = []

        def runtime(kernel):
            runtime_data, function = original_runtime(kernel)
            observations.append(
                {
                    **runtime_data,
                    "hooks": {
                        "reset_to_zero": list(kernel.reset_to_zero),
                        "restore_value": list(kernel.restore_value),
                        "user_defined_pre_hook": kernel.user_defined_pre_hook,
                        "user_defined_post_hook": kernel.user_defined_post_hook,
                    },
                }
            )
            return runtime_data, function

        _capture_module._kernel_runtime = runtime
        try:
            yield observations
        finally:
            _capture_module._kernel_runtime = original_runtime


def _load_art_module(root: Path) -> types.ModuleType:
    # The pinned file contributes three bare JITFunction objects.  The
    # Autotuner->JITFunction chain is admitted only from the live HOP callback.
    target = _target_module()
    target._validate_art_tree(root)
    for name in tuple(sys.modules):
        if name == "art" or name.startswith("art."):
            del sys.modules[name]
    package = types.ModuleType("art")
    package.__path__ = []
    package.__package__ = "art"
    trainer_rank = types.ModuleType("art.trainer_rank")
    trainer_rank.__path__ = []
    trainer_rank.__package__ = "art.trainer_rank"
    sys.modules["art"] = package
    sys.modules["art.trainer_rank"] = trainer_rank
    loaded = False
    try:
        module = _load_exact_module(
            "art.trainer_rank.topk",
            root / target.ART_SOURCE_PATH,
            target.ART_SOURCE_SHA256,
            "ART",
        )
        _validate_art_kernel_objects(module)
        loaded = True
        return module
    except (ImportError, OSError, SyntaxError) as exc:
        raise TargetError(f"could not load pinned ART source: {exc}") from exc
    finally:
        if not loaded:
            for name in tuple(sys.modules):
                if name == "art" or name.startswith("art."):
                    del sys.modules[name]


def _validate_art_kernel_objects(module: types.ModuleType) -> None:
    target = _target_module()
    try:
        from triton.runtime.jit import JITFunction
    except (ImportError, OSError) as exc:
        raise TargetError("the ART adapter requires Triton JIT support") from exc

    for name in target._ART_KERNEL_NAMES:
        kernel = getattr(module, name, None)
        if type(kernel) is not JITFunction:
            raise TargetError(f"ART source binding {name} is not an exact Triton JITFunction")


def _validate_art_runtime_config(config: dict[str, Any], label: str = "ART runtime Config") -> None:
    target = _target_module()
    expected = {
        "kwargs": {},
        "num_warps": target.ART_NUM_WARPS,
        "num_ctas": 1,
        "num_stages": 3,
        "maxnreg": None,
        "pre_hook": None,
        "ir_override": None,
    }
    if config != expected:
        raise InvalidArtifact(f"{label} is outside the exact ART runtime profile: {config!r}")


def _validate_art_audit_capture(
    candidate: Capture, runtime_observations: list[dict[str, Any]]
) -> None:
    try:
        _validate_art_capture_specialization(candidate)
        _validate_art_runtime_bindings(candidate, runtime_observations)
    except (InvalidArtifact, UnsupportedCapture, KeyError, TypeError, ValueError, AttributeError,
            IndexError, StopIteration) as exc:
        raise TargetError(f"ART capture could not be bound to the fixed target: {exc}") from exc


def _validate_art_runtime_bindings(
    capture: Capture,
    observations: list[dict[str, Any]],
) -> None:
    target = _target_module()
    if len(capture.kernels) != len(target._ART_KERNEL_NAMES) or len(observations) != len(target._ART_KERNEL_NAMES):
        raise UnsupportedCapture("ART runtime observation multiplicity changed")
    if {kernel.module for kernel in capture.kernels} != {target._ART_KERNEL_MODULE}:
        raise UnsupportedCapture("ART capture kernel module binding changed")
    if {kernel.qualname for kernel in capture.kernels} != set(target._ART_KERNEL_NAMES):
        raise UnsupportedCapture("ART capture kernel multiplicity or names changed")
    observed_by_key = {}
    for observation in observations:
        if not isinstance(observation, dict) or set(observation) != {
            "wrapper_type",
            "leaf_type",
            "config",
            "jit_cache_key",
            "hooks",
        }:
            raise UnsupportedCapture("ART runtime observation fields changed")
        key = observation["jit_cache_key"]
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(character not in "0123456789abcdef" for character in key)
            or key in observed_by_key
        ):
            raise UnsupportedCapture("ART runtime observation cache keys are not distinct")
        observed_by_key[key] = observation
    if len({kernel.runtime["jit_cache_key"] for kernel in capture.kernels}) != len(capture.kernels):
        raise UnsupportedCapture("ART capture kernel cache keys are not distinct")
    for kernel in capture.kernels:
        observation = observed_by_key.get(kernel.runtime.get("jit_cache_key"))
        if observation is None or observation["wrapper_type"] != kernel.runtime["wrapper_type"]:
            raise UnsupportedCapture("ART runtime wrapper observation changed")
        if observation["leaf_type"] != kernel.runtime["leaf_type"]:
            raise UnsupportedCapture("ART runtime leaf observation changed")
        if observation["config"] != kernel.runtime["config"]:
            raise UnsupportedCapture("ART runtime Config observation changed")
        if observation["hooks"] != {
            "reset_to_zero": [],
            "restore_value": [],
            "user_defined_pre_hook": False,
            "user_defined_post_hook": False,
        }:
            raise UnsupportedCapture("ART runtime hooks are outside the admitted profile")
        _validate_art_runtime_config(kernel.runtime["config"], f"ART kernel {kernel.qualname} Config")
    hops = [
        node
        for graph in (capture.forward, capture.backward)
        for node in graph.nodes
        if node.target == "triton.functional"
    ]
    if len(hops) != len(target._ART_KERNEL_NAMES):
        raise UnsupportedCapture("ART capture HOP multiplicity changed")
    kernels = {kernel.id: kernel for kernel in capture.kernels}
    hop_names = []
    for node in hops:
        if not isinstance(node.kwargs, dict):
            raise UnsupportedCapture("ART capture HOP fields changed")
        kernel = kernels.get(node.kwargs.get("kernel"))
        if kernel is None:
            raise UnsupportedCapture("ART capture HOP kernel binding changed")
        hop_names.append(kernel.qualname)
        if node.kwargs.get("grid") != _expected_art_grid(kernel.qualname):
            raise UnsupportedCapture("ART capture HOP grid changed")
    if len(set(hop_names)) != len(target._ART_KERNEL_NAMES) or set(hop_names) != set(target._ART_KERNEL_NAMES):
        raise UnsupportedCapture("ART capture HOP kernel multiplicity changed")


def _validate_art_capture_specialization(capture: Capture) -> None:
    target = _target_module()
    diagnostics = capture.diagnostics
    inputs = diagnostics.get("inputs") if isinstance(diagnostics, dict) else None
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise UnsupportedCapture("ART input specialization changed")
    device = inputs[0].get("device") if isinstance(inputs[0], dict) else None
    suffix = device[5:] if isinstance(device, str) and device.startswith("cuda:") else ""
    if not isinstance(device, str) or (device != "cuda" and (
        not suffix.isascii() or not suffix.isdigit() or str(int(suffix)) != suffix
    )):
        raise UnsupportedCapture("ART capture device specialization changed")
    rows, k = target.ART_SHAPE[0], target.ART_K
    pair, values, local = (rows, k), (rows, k, k), (rows,)
    layouts = {target.ART_SHAPE: (target.ART_SHAPE[1], 1), pair: (k, 1), values: (k * k, k, 1), local: (1,)}
    def ref(value):
        return _ref_id(value, "ART target node reference")

    def tensor(node, shape, dtype):
        value = node.tensor if node is not None else None
        return value is not None and (
            value.shape, value.stride, value.storage_offset, value.dtype, value.device
        ) == (shape, layouts[shape], 0, dtype, device)
    def runtime(value, shape, requires_grad):
        return isinstance(value, dict) and (
            tuple(value.get("shape", ())), tuple(value.get("stride", ())),
            value.get("storage_offset"), value.get("dtype"),
            value.get("device"), value.get("requires_grad")
        ) == (shape, layouts[shape], 0, "float32", device, requires_grad)
    def getitem(graph, hop, key):
        matches = [
            node for node in graph.nodes
            if node.target == "operator.getitem"
            and isinstance(node.args, dict)
            and node.args.get("tuple") == [{"ref": hop.id}, key]
        ]
        return matches[0] if len(matches) == 1 else None
    def output_refs(values, expected):
        ids = [node.id for node in expected if node is not None]
        return len(values) == len(expected) and ids == [ref(value) for value in values]
    def allocation(node, destination, shape, bindings, dtype):
        kind, source_role = destination
        source = ref(bindings[source_role]) if source_role else None
        args = {"tuple": [{"ref": source}]} if source else {"tuple": [list(shape)]}
        kwargs = {"pin_memory": False} if kind == "empty_like" else {
            "device": {"torch": device}, "dtype": {"torch": f"torch.{dtype}"},
            **({"layout": {"torch": "torch.strided"}}
               if kind == "zeros" else {"pin_memory": False}),
        }
        return node is not None and node.target == (
            "aten.empty_like" if kind == "empty_like" else f"aten.{kind}"
        ) and node.args == args and node.kwargs == kwargs
    if not runtime(inputs[0], target.ART_SHAPE, True) or not runtime(
        diagnostics.get("grad_output"), (target.ART_SHAPE[0],), False
    ):
        raise UnsupportedCapture("ART input metadata specialization changed")
    def f32(shape, destination=None):
        return shape, "float32", destination

    def i64(shape, destination=None):
        return shape, "int64", destination
    specs = {
        "_stats_stage1_kernel": (
            "forward", {"block_v": target.ART_BLOCK_V, "k": k, "n_blocks": 2,
                        "stride_row": target.ART_SHAPE[1], "vocab_size": target.ART_SHAPE[1]},
            {"logits_ptr": f32(target.ART_SHAPE), "partial_max_ptr": f32(pair, ("empty", None)),
             "partial_sum_ptr": f32(pair, ("empty_like", "partial_max_ptr")),
             "partial_tokens_ptr": i64(values, ("empty", None)), "partial_values_ptr": f32(values, ("empty", None))},
        ),
        "_stats_stage2_kernel": (
            "forward", {"block_b": 2, "block_candidates": 4, "k": k, "n_blocks": 2},
            {"local_max_ptr": f32(local, ("empty", None)),
             "local_sum_ptr": f32(local, ("empty_like", "local_max_ptr")),
             "partial_max_ptr": f32(pair), "partial_sum_ptr": f32(pair),
             "partial_tokens_ptr": i64(values), "partial_values_ptr": f32(values),
             "tokens_ptr": i64(pair, ("empty", None)), "values_ptr": f32(pair, ("empty", None))},
        ),
        "_stats_backward_kernel": (
            "backward", {"block_v": target.ART_BLOCK_V, "k": k,
                         "stride_row": target.ART_SHAPE[1], "vocab_size": target.ART_SHAPE[1]},
            {"grad_logits_ptr": f32(target.ART_SHAPE, ("empty_like", "logits_ptr")),
             "grad_sum_ptr": f32(local), "grad_values_ptr": f32(pair, ("zeros", None)),
             "local_max_ptr": f32(local), "logits_ptr": f32(target.ART_SHAPE), "tokens_ptr": i64(pair)},
        ),
    }
    kernels = {kernel.id: kernel for kernel in capture.kernels}
    graphs = {"forward": capture.forward, "backward": capture.backward}
    if (
        capture.forward.output_roles
        != (
            {"kind": "output", "index": 0},
            {"kind": "saved_output", "index": 0},
            {"kind": "saved_output", "index": 1},
            {"kind": "saved_output", "index": 2},
        )
        or capture.backward.output_roles
        != ({"kind": "gradient", "primal": {"kind": "primal", "index": 0}},)
    ):
        raise UnsupportedCapture("ART graph output roles changed")
    for graph in graphs.values():
        if any(node.tensor is not None and node.tensor.device != device for node in graph.nodes):
            raise UnsupportedCapture("ART graph tensor devices differ from its inputs")
    hops = {}
    for name, (graph_name, constants, bindings) in specs.items():
        graph = graphs[graph_name]
        hop = next(
            (
                node
                for node in graph.nodes
                if node.target == "triton.functional"
                and isinstance(node.kwargs, dict)
                and kernels.get(node.kwargs.get("kernel")) is not None
                and kernels[node.kwargs["kernel"]].qualname == name
            ),
            None,
        )
        if hop is None:
            raise UnsupportedCapture(f"ART {name} HOP is missing")
        nodes = {node.id: node for node in graph.nodes}
        if hop.kwargs.get("constants") != constants:
            raise UnsupportedCapture(f"ART {name} launch specialization changed")
        actual = hop.kwargs.get("bindings")
        if not isinstance(actual, dict) or set(actual) != set(bindings):
            raise UnsupportedCapture(f"ART {name} bindings changed")
        for role, (shape, dtype, destination) in bindings.items():
            node = nodes.get(ref(actual[role]))
            if not tensor(node, shape, dtype) or (
                destination and not allocation(node, destination, shape, actual, dtype)
            ):
                raise UnsupportedCapture(f"ART {name} destination {role} changed")
        hops[name] = (graph, hop, nodes)
    # Stage2 values_ptr has no getitem; its allocation is the checked result.
    stage1_graph, stage1, stage1_nodes = hops["_stats_stage1_kernel"]
    stage2_graph, stage2, _ = hops["_stats_stage2_kernel"]
    for role in ("partial_max_ptr", "partial_sum_ptr", "partial_tokens_ptr", "partial_values_ptr"):
        result = getitem(stage1_graph, stage1, role)
        if result is None or ref(stage2.kwargs["bindings"][role]) != result.id:
            raise UnsupportedCapture("ART stage-two partial binding changed")
    forward, backward = capture.forward, capture.backward
    forward_values = _graph_output_refs(forward.nodes[-1].args)
    primal = stage1_nodes.get(ref(stage1.kwargs["bindings"]["logits_ptr"]))
    if (
        primal is None
        or (primal.op, primal.target) != ("placeholder", "placeholder")
        or primal.role != {"kind": "primal", "index": 0}
    ):
        raise UnsupportedCapture("ART forward primal role changed")
    local_sum = getitem(stage2_graph, stage2, "local_sum_ptr")
    local_max = getitem(stage2_graph, stage2, "local_max_ptr")
    tokens = getitem(stage2_graph, stage2, "tokens_ptr")
    if (
        not output_refs(forward_values, (local_sum, primal, local_max, tokens))
        or not tensor(local_sum, local, "float32")
        or not tensor(local_max, local, "float32")
        or not tensor(tokens, pair, "int64")
    ):
        raise UnsupportedCapture("ART forward output semantics changed")
    backward_values = _graph_output_refs(backward.nodes[-1].args)
    backward_graph, backward_hop, backward_nodes = hops["_stats_backward_kernel"]
    backward_bindings = backward_hop.kwargs["bindings"]
    saved_local_sum = backward_nodes.get(ref(backward_bindings["local_max_ptr"]))
    saved_tokens = backward_nodes.get(ref(backward_bindings["tokens_ptr"]))
    tangent = backward_nodes.get(ref(backward_bindings["grad_sum_ptr"]))
    backward_primal = backward_nodes.get(ref(backward_bindings["logits_ptr"]))
    if (
        backward_primal is None
        or (backward_primal.op, backward_primal.target) != ("placeholder", "placeholder")
        or backward_primal.role != {"kind": "primal", "index": 0}
        or any(
            node is None
            or (node.op, node.target) != ("placeholder", "placeholder")
            or node.role is not None
            for node in (saved_local_sum, saved_tokens)
        )
        or tangent is None
        or (tangent.op, tangent.target) != ("placeholder", "placeholder")
        or tangent.role != {"kind": "tangent", "output": {"kind": "output", "index": 0}}
    ):
        raise UnsupportedCapture("ART backward saved-input roles changed")
    grad_logits = getitem(backward_graph, backward_hop, "grad_logits_ptr")
    if (
        not output_refs(backward_values, (grad_logits,))
        or not tensor(grad_logits, target.ART_SHAPE, "float32")
    ):
        raise UnsupportedCapture("ART backward output semantics changed")


def _expected_art_grid(qualname: str) -> list[dict[str, list[int]]]:
    target = _target_module()
    rows = target.ART_SHAPE[0]
    n_blocks = (target.ART_SHAPE[1] + target.ART_BLOCK_V - 1) // target.ART_BLOCK_V
    if qualname == "_stats_stage2_kernel":
        return [{"tuple": [rows, 1, 1]}]
    if qualname in {"_stats_stage1_kernel", "_stats_backward_kernel"}:
        return [{"tuple": [rows, n_blocks, 1]}]
    raise UnsupportedCapture(f"ART kernel {qualname} has no admitted grid specialization")


def _observed_art_grids(capture: Capture) -> list[dict[str, Any]]:
    kernels = {kernel.id: kernel for kernel in capture.kernels}
    observed = []
    for graph in (capture.forward, capture.backward):
        for node in graph.nodes:
            if node.target != "triton.functional" or not isinstance(node.kwargs, dict):
                continue
            kernel = kernels.get(node.kwargs.get("kernel"))
            if kernel is None:
                raise UnsupportedCapture("ART HOP grid references an unknown kernel")
            observed.append(
                {
                    "graph": graph.kind,
                    "qualname": kernel.qualname,
                    "grid": node.kwargs["grid"],
                }
            )
    return observed
