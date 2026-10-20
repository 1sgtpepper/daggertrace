from __future__ import annotations

import inspect
import math
import operator
from collections.abc import Callable, Sequence
from typing import Any

from .canonical import sha256_bytes, sha256_json
from .effects import profile_triton_source
from .errors import InvalidArtifact, UnsupportedCapture
from .model import Capture, Graph, Kernel, Node, TensorMeta, ref
from .model_core import _is_closed_config


_SUPPORTED_TORCH = "2.12.1+cu130"
_SUPPORTED_TORCH_GIT = "7269437d655783a26cba32aa88195b741ff496aa"
_SUPPORTED_TRITON = "3.7.1"


def capture_aot(
    function: Callable[..., Any],
    inputs: Sequence[Any],
    grad_output: Any,
    *,
    label: str,
) -> Capture:
    """Capture structured forward/backward AOT graphs before Inductor lowering.

    Dynamo is required here: direct ``aot_function`` reaches the raw kernel launch
    before it has been converted to a traceable Triton higher-order operator.
    """

    import torch
    import triton

    _require_framework(torch.__version__, torch.version.git_version, triton.__version__)
    try:
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd
    except (ImportError, AttributeError) as exc:
        raise UnsupportedCapture("required private AOT capture APIs are unavailable") from exc
    torch._dynamo.reset()
    counter_before = dict(torch._dynamo.utils.counters["aot_autograd"])
    captured: dict[str, Graph] = {}
    kernels: dict[str, Kernel] = {}
    callbacks = {"forward": 0, "backward": 0}

    def recorder(kind: str):
        def compile_graph(graph_module, _example_inputs):
            callbacks[kind] += 1
            if callbacks[kind] != 1:
                raise UnsupportedCapture(f"expected one {kind} graph, observed {callbacks[kind]}")
            captured[kind] = _normalize_graph(graph_module, kind, kernels)
            return make_boxed_func(graph_module.forward)

        return compile_graph

    backend = aot_autograd(fw_compiler=recorder("forward"), bw_compiler=recorder("backward"))
    try:
        compiled = torch.compile(function, fullgraph=True, dynamic=False, backend=backend)
        output = compiled(*inputs)
        differentiable = tuple(value for value in inputs if getattr(value, "requires_grad", False))
        if not differentiable:
            raise UnsupportedCapture("capture needs at least one differentiable input")
        torch.autograd.grad(output, differentiable, grad_outputs=grad_output)
    except torch._dynamo.exc.Unsupported as exc:
        raise UnsupportedCapture(f"Dynamo could not capture the exact graph: {exc}") from exc
    if set(captured) != {"forward", "backward"}:
        raise UnsupportedCapture(f"capture callbacks incomplete: {sorted(captured)}")

    framework = {
        "python": _python_version(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    diagnostics = {
        "callbacks": callbacks,
        "aot_autograd": _counter_delta(
            counter_before, dict(torch._dynamo.utils.counters["aot_autograd"])
        ),
        "inputs": [_runtime_tensor_meta(value) for value in inputs],
        "grad_output": _runtime_tensor_meta(grad_output),
    }
    capture = Capture(
        label=label,
        framework=framework,
        capture_config={
            "fullgraph": True,
            "dynamic": False,
            "stage": "post-dynamo-pre-inductor",
        },
        forward=captured["forward"],
        backward=captured["backward"],
        kernels=tuple(sorted(kernels.values(), key=lambda kernel: kernel.id)),
        diagnostics=diagnostics,
    )
    try:
        return Capture.from_dict(capture)
    except InvalidArtifact as exc:
        raise UnsupportedCapture("captured result is not a valid v2 capture") from exc


def _normalize_graph(graph_module: Any, kind: str, kernels: dict[str, Kernel]) -> Graph:
    fx_nodes = list(graph_module.graph.nodes)
    node_ids = {node: f"n{index}" for index, node in enumerate(fx_nodes)}
    alias_groups: dict[tuple[str, int], str] = {}
    nodes: list[Node] = []
    output_roles: tuple[dict[str, Any], ...] = ()
    for fx_node in fx_nodes:
        target = _target(fx_node)
        if target == "triton.functional":
            kwargs = _normalize_hop(fx_node, node_ids, kernels)
            args: Any = []
        else:
            args = _value(fx_node.args, node_ids)
            kwargs = _value(fx_node.kwargs, node_ids)
        role = _descriptor(fx_node.meta.get("desc")) if fx_node.op == "placeholder" else None
        if role is not None and not isinstance(role, dict):
            raise UnsupportedCapture("AOT placeholder descriptor is not a role")
        if fx_node.op == "output":
            described = _descriptor(fx_node.meta.get("desc"))
            if not isinstance(described, list) or not all(
                isinstance(item, dict) for item in described
            ):
                raise UnsupportedCapture("AOT output descriptor is not a role list")
            output_roles = tuple(described)
        nodes.append(
            Node(
                id=node_ids[fx_node],
                op=fx_node.op,
                target=target,
                args=args,
                kwargs=kwargs,
                role=role,
                tensor=_tensor_meta(fx_node, alias_groups),
                gradient_accumulation=(
                    target == "aten.add" and fx_node.meta.get("is_gradient_acc") is True
                ),
            )
        )
    return Graph(kind=kind, nodes=tuple(nodes), output_roles=output_roles)


def _normalize_hop(fx_node: Any, node_ids: dict[Any, str], kernels: dict[str, Kernel]) -> dict[str, Any]:
    from torch._higher_order_ops.triton_kernel_wrap import (
        get_kernel,
        identify_accessed_tensors,
        kernel_side_table,
    )

    raw = dict(fx_node.kwargs)
    expected = {"kernel_idx", "constant_args_idx", "grid", "tma_descriptor_metadata", "kwargs", "tensors_to_clone"}
    if set(raw) != expected:
        raise UnsupportedCapture(f"functional Triton HOP fields changed: {sorted(raw)}")
    if raw["tma_descriptor_metadata"]:
        raise UnsupportedCapture("TMA descriptors are outside the ownership profile")
    bindings, inline_constants = _partition_hop_arguments(raw["kwargs"], node_ids)
    writes = tuple(sorted(raw["tensors_to_clone"]))
    if not writes:
        raise UnsupportedCapture("functional Triton HOP has no write roles")
    if not set(writes) <= set(bindings):
        raise UnsupportedCapture("functional Triton HOP write is not a tensor binding")
    kernel_object = get_kernel(raw["kernel_idx"])
    side_constants = kernel_side_table.get_constant_args(raw["constant_args_idx"])
    if not isinstance(side_constants, dict) or not all(
        isinstance(name, str) for name in side_constants
    ):
        raise UnsupportedCapture("Triton HOP side-table constants changed")
    if set(side_constants) & set(inline_constants):
        raise UnsupportedCapture("Triton HOP constant is present in two owners")
    constants = {**side_constants, **inline_constants}
    runtime, python_function = _kernel_runtime(kernel_object)
    source = inspect.getsource(python_function)
    source_sha = sha256_bytes(source.encode("utf-8"))

    eager_values = fx_node.meta.get("eager_input_vals")
    if not isinstance(eager_values, tuple) or len(eager_values) != 2 or not isinstance(eager_values[1], dict):
        raise UnsupportedCapture("functional Triton HOP lacks eager input metadata")
    fake_kwargs = eager_values[1].get("kwargs")
    if not isinstance(fake_kwargs, dict) or set(fake_kwargs) != set(bindings) | set(
        inline_constants
    ):
        raise UnsupportedCapture("functional Triton HOP eager bindings changed")
    accesses = identify_accessed_tensors(
        kernel_object,
        {**fake_kwargs, **constants},
        raw["tma_descriptor_metadata"],
    )
    reads = tuple(sorted(dependency.name for dependency in accesses.read_writes.reads))
    ttir_writes = tuple(sorted(dependency.name for dependency in accesses.read_writes.writes))
    if ttir_writes != writes:
        raise UnsupportedCapture(
            f"TTIR writes {ttir_writes} disagree with functional writes {writes}; effect fallback is not proof"
        )
    source_profile = profile_triton_source(source, set(bindings))
    if source_profile.reads != reads or source_profile.writes != writes:
        raise UnsupportedCapture(
            "source and TTIR effects disagree: "
            f"source=({source_profile.reads},{source_profile.writes}), ttir=({reads},{writes})"
        )
    identity = {
        "module": python_function.__module__,
        "qualname": python_function.__qualname__,
        "source_sha256": source_sha,
        "arguments": list(source_profile.arguments),
        "reads": list(reads),
        "writes": list(writes),
        "runtime": runtime,
    }
    kernel_id = sha256_json(identity)
    kernels.setdefault(
        kernel_id,
        Kernel(
            id=kernel_id,
            module=python_function.__module__,
            qualname=python_function.__qualname__,
            source_sha256=source_sha,
            source=source,
            argument_names=source_profile.arguments,
            reads=reads,
            writes=writes,
            source_reads=source_profile.reads,
            source_writes=source_profile.writes,
            runtime=runtime,
        ),
    )
    return {
        "kernel": kernel_id,
        "bindings": {name: ref(node_ids[value]) for name, value in sorted(bindings.items())},
        "writes": list(writes),
        "constants": _value(constants, node_ids),
        "grid": _value(raw["grid"], node_ids),
    }


def _partition_hop_arguments(
    values: Any,
    node_ids: dict[Any, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(values, dict) or not values or not all(
        isinstance(name, str) for name in values
    ):
        raise UnsupportedCapture("Triton HOP arguments are absent or malformed")
    bindings: dict[str, Any] = {}
    constants: dict[str, Any] = {}
    for name, value in values.items():
        try:
            is_node = value in node_ids
        except TypeError:
            is_node = False
        if is_node:
            bindings[name] = value
        else:
            constants[name] = _json_runtime_value(value, f"Triton HOP constant {name}")
    if not bindings:
        raise UnsupportedCapture("Triton HOP tensor bindings are absent")
    return bindings, constants


def _kernel_runtime(kernel: Any) -> tuple[dict[str, Any], Callable[..., Any]]:
    from triton.runtime.autotuner import Autotuner, Config
    from triton.runtime.jit import JITFunction

    if type(kernel) is not Autotuner or type(kernel.fn) is not JITFunction:
        raise UnsupportedCapture(
            "Triton runtime wrapper must be exactly Autotuner -> JITFunction"
        )
    if len(kernel.configs) != 1 or type(kernel.configs[0]) is not Config:
        raise UnsupportedCapture("Triton Autotuner must contain exactly one exact Config")
    if (
        kernel.reset_to_zero != []
        or kernel.restore_value != []
        or kernel.user_defined_pre_hook is not False
        or kernel.user_defined_post_hook is not False
    ):
        raise UnsupportedCapture("Triton Autotuner hooks/reset/restore are outside the profile")

    config = vars(kernel.configs[0])
    config_fields = {
        "kwargs",
        "num_warps",
        "num_ctas",
        "num_stages",
        "maxnreg",
        "pre_hook",
        "ir_override",
    }
    if set(config) != config_fields:
        raise UnsupportedCapture(f"Triton Config fields changed: {sorted(config)}")
    if not _is_closed_config(config):
        raise UnsupportedCapture("Triton Config is outside the exact admitted profile")
    normalized_config = _json_runtime_value(config, "Triton Config")
    cache_key = kernel.fn.cache_key
    if (
        not isinstance(cache_key, str)
        or len(cache_key) != 64
        or any(character not in "0123456789abcdef" for character in cache_key)
    ):
        raise UnsupportedCapture("Triton JITFunction cache key is not a lowercase SHA-256 digest")
    return (
        {
            "wrapper_type": f"{Autotuner.__module__}.{Autotuner.__qualname__}",
            "leaf_type": f"{JITFunction.__module__}.{JITFunction.__qualname__}",
            "config": normalized_config,
            "jit_cache_key": cache_key,
        },
        kernel.fn.fn,
    )


def _json_runtime_value(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UnsupportedCapture(f"{label} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_json_runtime_value(item, f"{label}[]") for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise UnsupportedCapture(f"{label} contains a non-string key")
        return {
            key: _json_runtime_value(item, f"{label}.{key}")
            for key, item in sorted(value.items())
        }
    raise UnsupportedCapture(
        f"{label} contains unsupported value {type(value).__module__}.{type(value).__qualname__}"
    )


def _target(node: Any) -> str:
    import torch
    from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_functional

    if node.op == "placeholder":
        return "placeholder"
    if node.op == "output":
        return "output"
    if node.target is operator.getitem:
        return "operator.getitem"
    mapping = {
        torch.ops.aten.view.default: "aten.view",
        torch.ops.aten._unsafe_view.default: "aten._unsafe_view",
        torch.ops.aten.alias.default: "aten.alias",
        torch.ops.aten.detach.default: "aten.detach",
        torch.ops.aten.empty_like.default: "aten.empty_like",
        torch.ops.aten.zeros_like.default: "aten.zeros_like",
        torch.ops.aten.empty.memory_format: "aten.empty",
        torch.ops.aten.zeros.default: "aten.zeros",
        torch.ops.aten.clone.default: "aten.clone",
        torch.ops.aten.mul.Tensor: "aten.mul",
        torch.ops.aten.add.Tensor: "aten.add",
        torch.ops.aten.sum.dim_IntList: "aten.sum",
    }
    if node.target in mapping:
        return mapping[node.target]
    if node.target is triton_kernel_wrapper_functional:
        return "triton.functional"
    return f"unsupported:{type(node.target).__module__}.{type(node.target).__qualname__}"


def _value(value: Any, node_ids: dict[Any, str]) -> Any:
    import torch
    import triton.language as tl

    try:
        if value in node_ids:
            return ref(node_ids[value])
    except (TypeError, RuntimeError):
        pass
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return {"tuple": [_value(item, node_ids) for item in value]}
    if isinstance(value, list):
        return [_value(item, node_ids) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise UnsupportedCapture("captured dictionaries require string keys")
        return {key: _value(item, node_ids) for key, item in sorted(value.items())}
    if isinstance(value, (torch.device, torch.dtype, torch.layout, torch.memory_format)):
        return {"torch": str(value)}
    if isinstance(value, tl.dtype):
        return {"torch": str(value)}
    raise UnsupportedCapture(f"unsupported captured literal {type(value).__module__}.{type(value).__name__}")


def _descriptor(value: Any) -> Any:
    from torch._functorch._aot_autograd.descriptors import (
        GradAOTOutput,
        PlainAOTInput,
        PlainAOTOutput,
        SavedForBackwardsAOTOutput,
        TangentAOTInput,
    )

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_descriptor(item) for item in value]
    if type(value) is PlainAOTInput:
        return {"kind": "primal", "index": int(value.idx)}
    if type(value) is PlainAOTOutput:
        return {"kind": "output", "index": int(value.idx)}
    if type(value) is TangentAOTInput:
        return {"kind": "tangent", "output": _descriptor(value.output)}
    if type(value) is GradAOTOutput:
        return {"kind": "gradient", "primal": _descriptor(value.grad_of)}
    if type(value) is SavedForBackwardsAOTOutput:
        return {"kind": "saved_output", "index": int(value.idx)}
    return {
        "kind": "unsupported",
        "descriptor": f"{type(value).__module__}.{type(value).__qualname__}",
    }


def _tensor_meta(node: Any, alias_groups: dict[tuple[str, int], str]) -> TensorMeta | None:
    metadata = node.meta.get("tensor_meta")
    value = node.meta.get("val")
    if metadata is None or not hasattr(metadata, "shape") or not hasattr(metadata, "stride"):
        return None
    device = str(getattr(value, "device", "unknown"))
    storage_offset = getattr(value, "storage_offset", lambda: 0)()
    alias_group = None
    try:
        storage = value.untyped_storage()
        storage_key = (str(value.device), int(storage._cdata))
        alias_group = alias_groups.setdefault(storage_key, f"a{len(alias_groups)}")
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return TensorMeta(
        shape=tuple(int(dimension) for dimension in metadata.shape),
        stride=tuple(int(stride) for stride in metadata.stride),
        dtype=str(metadata.dtype).removeprefix("torch."),
        device=device,
        storage_offset=int(storage_offset),
        alias_group=alias_group,
    )


def _runtime_tensor_meta(value: Any) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
    }


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in sorted(set(before) | set(after))
        if after.get(key, 0) != before.get(key, 0)
    }


def _require_framework(torch_version: str, torch_git: str | None, triton_version: str) -> None:
    if torch_version != _SUPPORTED_TORCH or torch_git != _SUPPORTED_TORCH_GIT or triton_version != _SUPPORTED_TRITON:
        raise UnsupportedCapture(
            "framework profile mismatch: "
            f"torch={torch_version}@{torch_git}, triton={triton_version}; "
            f"expected torch={_SUPPORTED_TORCH}@{_SUPPORTED_TORCH_GIT}, triton={_SUPPORTED_TRITON}"
        )


def _python_version() -> str:
    import platform

    return platform.python_version()
