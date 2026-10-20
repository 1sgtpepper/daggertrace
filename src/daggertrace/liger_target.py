"""Audit, numerical comparisons, and replay checks for Liger."""

from __future__ import annotations

import math
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any

from .canonical import sha256_bytes, write_canonical_json
from .capture import capture_aot
from .errors import InvalidArtifact, TargetError, UnsupportedCapture
from .model import Capture
from . import liger_capture as _liger_capture
from . import source_files as _source_files
from .liger_packet import (
    LIGER_REPOSITORY,
    LIGER_REVISION,
    LIGER_ROOT_UTILS_PATH,
    LIGER_ROOT_UTILS_SHA256,
    LIGER_SOURCE_PATH,
    LIGER_SOURCE_SHA256,
    LIGER_UTILS_PATH,
    LIGER_UTILS_SHA256,
    _write_reproducer,
)
from . import ownership as _ownership
from . import liger_packet as _liger_packet
from .target_support import (
    _counter_delta,
    _counter_snapshot,
    _fetch_pinned_archive,
    _gpu_environment,
    _max_abs,
    _require_exact_gpu_framework,
    _require_generic_target_result,
    _require_module,
    _tensor_hash,
    _validate_benchmark_bounds,
)
SHAPE = (2, 128)
SEED = 7

def fetch_liger(cache_dir: Path) -> Path:
    return _fetch_pinned_archive(
        cache_dir,
        root_name=f"Liger-Kernel-{LIGER_REVISION}",
        repository=LIGER_REPOSITORY,
        revision=LIGER_REVISION,
        user_agent="daggertrace/0.4",
        label="Liger",
        validate_tree=_liger_packet._validate_liger_tree,
    )


def audit_liger(
    output_dir: Path,
    *,
    target_root: Path | None = None,
    warmups: int = 3,
    samples: int = 10,
) -> dict[str, Any]:
    """Run the complete source/capture/replay-ready Liger demonstration."""

    _validate_benchmark_bounds(warmups, samples)
    torch = _require_module("torch", "PyTorch")
    triton = _require_module("triton", "Triton")

    if not torch.cuda.is_available():
        raise TargetError("the Liger demonstration requires CUDA")
    _require_exact_gpu_framework(torch, triton, "the Liger demonstration")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        root = (
            target_root
            if target_root is not None
            else fetch_liger(output_dir.parent / ".target-cache")
        )
    except OSError as exc:
        raise TargetError(f"could not prepare the Liger audit paths: {exc}") from exc
    function_class = _load_liger_function(root)
    torch.manual_seed(SEED)
    x0 = torch.randn(SHAPE, device="cuda", dtype=torch.float32)
    weight0 = torch.randn(SHAPE[-1], device="cuda", dtype=torch.float32)
    cotangent0 = torch.randn(SHAPE, device="cuda", dtype=torch.float32)

    direct_candidate_fn = _liger_function(function_class, "direct", True)
    direct_control_fn = _liger_function(function_class, "direct", False)
    candidate = _capture(direct_candidate_fn, x0, weight0, cotangent0, "liger-direct-in-place")
    control = _capture(direct_control_fn, x0, weight0, cotangent0, "liger-direct-out-of-place")
    _validate_liger_capture_specialization(candidate, control)
    analysis = _ownership.analyze(candidate, control)
    if analysis.outcome.value != "VIOLATION" or analysis.collapse is None:
        raise TargetError(f"expected a borrowed write with contribution collapse: {analysis.to_dict()}")

    reference_fn = _reference_function()
    runtime, target_runtime, compiled_functions = _runtime_evidence_pair(
        reference_fn,
        direct_candidate_fn,
        direct_control_fn,
        x0,
        weight0,
        cotangent0,
    )
    controls, control_witness = _control_evidence(function_class, x0, weight0, cotangent0)
    target_runtime["control_witness"] = control_witness
    _validate_liger_runtime_witness(target_runtime)
    _validate_liger_capture_specialization(
        candidate,
        control,
        expected_partial_rows=target_runtime["environment"]["multiprocessors"],
    )
    benchmark = _benchmark(
        {
            "torch_reference": reference_fn,
            "liger_out_of_place": direct_control_fn,
            "liger_in_place": direct_candidate_fn,
            "compiled_out_of_place": compiled_functions["control"],
            "compiled_in_place": compiled_functions["candidate"],
        },
        x0,
        weight0,
        cotangent0,
        warmups=warmups,
        samples=samples,
    )
    try:
        _liger_packet._stage_external_sources(root, output_dir)
        target = _target_identity(
            root,
            candidate,
            output_dir,
            control=control,
            runtime_witness=target_runtime,
        )
        write_canonical_json(output_dir / "candidate.capture.json", candidate.to_dict())
        write_canonical_json(output_dir / "control.capture.json", control.to_dict())
        write_canonical_json(output_dir / "runtime.json", runtime)
        write_canonical_json(output_dir / "benchmark.json", benchmark)
        write_canonical_json(output_dir / "controls.json", controls)
        _write_reproducer(output_dir)
    except OSError as exc:
        raise TargetError(f"could not stage the Liger audit artifact: {exc}") from exc
    return {
        "target": target,
        "candidate": candidate,
        "control": control,
        "analysis": analysis,
        "runtime": runtime,
        "target_runtime": target_runtime,
        "benchmark": benchmark,
        "controls": controls,
    }


def run_liger_reproducer(target_root: Path | None = None) -> dict[str, Any]:
    """Run the minimal eager numerical reproducer."""

    torch = _require_module("torch", "PyTorch")
    triton = _require_module("triton", "Triton")
    if not torch.cuda.is_available():
        raise TargetError("the Liger reproducer requires CUDA")
    _require_exact_gpu_framework(torch, triton, "the Liger reproducer")

    try:
        root = (
            target_root
            if target_root is not None
            else fetch_liger(Path.cwd() / ".target-cache")
        )
    except OSError as exc:
        raise TargetError(f"could not prepare the Liger reproducer paths: {exc}") from exc
    function_class = _load_liger_function(root)
    torch.manual_seed(SEED)
    x0 = torch.randn(SHAPE, device="cuda", dtype=torch.float32)
    weight0 = torch.randn(SHAPE[-1], device="cuda", dtype=torch.float32)
    cotangent0 = torch.randn(SHAPE, device="cuda", dtype=torch.float32)
    reference = _run_lane(_reference_function(), x0, weight0, cotangent0)
    candidate = _run_lane(_liger_function(function_class, "direct", True), x0, weight0, cotangent0)
    control = _run_lane(_liger_function(function_class, "direct", False), x0, weight0, cotangent0)
    comparison = _comparison(reference, candidate, control)
    _validate_eager_evidence(comparison)
    return comparison


def _replay_liger_loaded(
    certificate: dict[str, Any],
    certificate_path: Path,
    generic_result: dict[str, Any],
) -> dict[str, Any]:
    """Validate Liger's target-owned specialization after generic replay."""

    try:
        _require_generic_target_result(
            generic_result,
            outcome="VIOLATION",
            reasons=["BORROWED_COTANGENT_WRITE", "CONTRIBUTION_COLLAPSE"],
            borrowed_writes=1,
            contextual_collapse=True,
            target="Liger",
        )
        packet = certificate_path.parent.resolve()
        if certificate_path.name != "certificate.json":
            raise TargetError("Liger packet certificate must be named certificate.json")
        target = certificate["target"]
        if (
            not isinstance(target, dict)
            or target.get("repository") != LIGER_REPOSITORY
            or target.get("revision") != LIGER_REVISION
        ):
            raise TargetError("Liger target repository or revision changed")
        if certificate.get("control") is None:
            raise TargetError("Liger replay requires a paired control capture")
        specialization = target.get("specialization")
        if not isinstance(specialization, dict):
            raise TargetError("Liger target specialization changed")
        runtime_witness = specialization.get("runtime_witness")
        if not isinstance(runtime_witness, dict):
            raise TargetError("Liger target runtime witness is missing")
        _validate_liger_runtime_witness(runtime_witness)
        if certificate["runtime"] != _generic_liger_runtime(runtime_witness):
            raise TargetError("Liger generic runtime lane does not match its target witness")
        candidate = Capture.from_dict(certificate["candidate"])
        control = Capture.from_dict(certificate["control"])
        _validate_liger_capture_specialization(
            candidate,
            control,
            expected_partial_rows=runtime_witness["environment"]["multiprocessors"],
        )
        tree = packet / "external" / "liger" / "tree"
        if target != _target_identity(
            tree,
            candidate,
            packet,
            control=control,
            runtime_witness=runtime_witness,
        ):
            raise TargetError("Liger target identity does not match supplied source/capture bytes")
        return {
            "adapter": "liger-rmsnorm",
            "adapter_only": False,
            "generic_replay_compatible": True,
            "replay": "generic-then-liger-target-closure",
            "remote_revision_authenticity": "not_proven",
        }
    except TargetError:
        raise
    except (
        InvalidArtifact,
        UnsupportedCapture,
        OSError,
        UnicodeError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        IndexError,
        RecursionError,
    ) as exc:
        raise TargetError(f"Liger packet could not be replayed: {exc}") from exc


def _validate_liger_capture_specialization(
    candidate: Capture,
    control: Capture,
    *,
    expected_partial_rows: int | None = None,
) -> None:
    return _liger_capture.validate_liger_capture_specialization(
        candidate,
        control,
        shape=SHAPE,
        expected_partial_rows=expected_partial_rows,
    )

def _capture(function, x0, weight0, cotangent0, label):
    x = x0.clone().requires_grad_(True)
    weight = weight0.clone().requires_grad_(True)
    return capture_aot(function, (x, weight), cotangent0.clone(), label=label)


def _runtime_evidence_pair(reference_fn, candidate_fn, control_fn, x0, weight0, cotangent0):
    reference = _run_lane(reference_fn, x0, weight0, cotangent0)
    candidate_eager = _run_lane(candidate_fn, x0, weight0, cotangent0)
    control_eager = _run_lane(control_fn, x0, weight0, cotangent0)
    compiled_candidate, candidate_compiled, candidate_compile = _compile_lane(
        candidate_fn, x0, weight0, cotangent0
    )
    compiled_control, control_compiled, control_compile = _compile_lane(
        control_fn, x0, weight0, cotangent0
    )
    comparison = _comparison(reference, candidate_eager, control_eager)
    comparison["compiled"] = {
        "candidate_vs_reference": _lane_delta(candidate_compiled, reference),
        "control_vs_reference": _lane_delta(control_compiled, reference),
        "candidate_eager_vs_compiled": _lane_delta(candidate_compiled, candidate_eager),
        "control_eager_vs_compiled": _lane_delta(control_compiled, control_eager),
        "candidate_observation": _lane_observation(candidate_compiled),
        "control_observation": _lane_observation(control_compiled),
        "compile": {"candidate": candidate_compile, "control": control_compile},
    }
    _validate_runtime_evidence(comparison)
    comparison["environment"] = _gpu_environment("Liger")
    comparison["shape"] = list(SHAPE)
    comparison["seed"] = SEED
    _validate_liger_runtime_witness(comparison, require_control_witness=False)
    return (
        _generic_liger_runtime(comparison),
        comparison,
        {"candidate": compiled_candidate, "control": compiled_control},
    )


def _generic_liger_runtime(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "environment": comparison["environment"],
        "candidate": {
            "compiled": {
                "occurrence": comparison["compiled"]["compile"]["candidate"]
            }
        },
        "control": {
            "compiled": {
                "occurrence": comparison["compiled"]["compile"]["control"]
            }
        },
    }


def _validate_runtime_evidence(comparison):
    _validate_eager_evidence(comparison)
    compiled = comparison["compiled"]
    candidate_reference = compiled["candidate_vs_reference"]
    control_reference = compiled["control_vs_reference"]
    if (
        not math.isfinite(candidate_reference["gx_max_abs"])
        or not candidate_reference["gx_max_abs"] > 1e-4
    ):
        raise TargetError("compiled in-place residual did not preserve the numerical witness")
    if (
        not 0.0 <= candidate_reference["output_max_abs"] <= 2e-5
        or not 0.0 <= candidate_reference["gw_max_abs"] <= 2e-5
    ):
        raise TargetError("compiled in-place lane changed the reference output or dW")
    if any(
        not 0.0 <= control_reference[key] <= 2e-5
        for key in ("output_max_abs", "gx_max_abs", "gw_max_abs")
    ):
        raise TargetError("compiled out-of-place control does not match the pure-Torch reference")
    for name in ("candidate_eager_vs_compiled", "control_eager_vs_compiled"):
        if any(value != 0.0 for value in compiled[name].values()):
            raise TargetError(f"{name} changed output or gradients")
    for name in ("candidate_observation", "control_observation"):
        observation = compiled[name]
        if observation["cotangent_delta"] != 0.0:
            raise TargetError(f"compiled {name} mutated the external cotangent")
        if observation["cotangent_version"] != [0, 0]:
            raise TargetError(f"compiled {name} changed the cotangent tensor version")


def _validate_liger_runtime_witness(
    comparison: dict[str, Any], *, require_control_witness: bool = True
) -> None:
    from .replay import ReplayError
    from .replay_packet import _closed, _environment, _integer, _is_digest, _number

    expected = {
        "candidate_vs_reference",
        "control_vs_reference",
        "candidate_vs_control",
        "cotangent",
        "peak_allocated_bytes",
        "hashes",
        "compiled",
        "environment",
        "shape",
        "seed",
    }
    has_control_witness = isinstance(comparison, dict) and "control_witness" in comparison
    if require_control_witness and not has_control_witness:
        raise TargetError("Liger target runtime witness control evidence is missing")
    if has_control_witness:
        expected.add("control_witness")
    if not isinstance(comparison, dict) or set(comparison) != expected:
        raise TargetError("Liger target runtime witness fields changed")
    if comparison["shape"] != list(SHAPE):
        raise TargetError("Liger target runtime witness shape changed")
    if comparison["seed"] != SEED:
        raise TargetError("Liger target runtime witness seed changed")
    try:
        _environment(comparison["environment"])

        def delta(raw: Any, label: str) -> None:
            values = _closed(raw, {"output_max_abs", "gx_max_abs", "gw_max_abs"}, label)
            for key, value in values.items():
                _number(value, f"{label}.{key}")

        def version(raw: Any, label: str) -> None:
            if not isinstance(raw, list) or len(raw) != 2:
                raise TargetError(f"{label} must contain two tensor versions")
            for index, value in enumerate(raw):
                _integer(value, f"{label}[{index}]", minimum=0)

        for name in (
            "candidate_vs_reference",
            "control_vs_reference",
            "candidate_vs_control",
        ):
            delta(comparison[name], f"Liger runtime {name}")
        cotangent = _closed(
            comparison["cotangent"],
            {"candidate_delta", "control_delta", "candidate_version", "control_version"},
            "Liger runtime cotangent",
        )
        _number(cotangent["candidate_delta"], "Liger runtime cotangent.candidate_delta")
        _number(cotangent["control_delta"], "Liger runtime cotangent.control_delta")
        version(cotangent["candidate_version"], "Liger runtime cotangent.candidate_version")
        version(cotangent["control_version"], "Liger runtime cotangent.control_version")
        peak = _closed(
            comparison["peak_allocated_bytes"],
            {"reference", "candidate", "control"},
            "Liger runtime peak_allocated_bytes",
        )
        for key, value in peak.items():
            _integer(value, f"Liger runtime peak_allocated_bytes.{key}", minimum=0)
        hashes = _closed(
            comparison["hashes"],
            {"reference_gx", "candidate_gx", "control_gx"},
            "Liger runtime hashes",
        )
        if not all(_is_digest(value) for value in hashes.values()):
            raise TargetError("Liger runtime gradient hash is malformed")
        if has_control_witness:
            control_witness = _closed(
                comparison["control_witness"],
                {"isolated", "scaled"},
                "Liger runtime control_witness",
            )
            for name, raw in control_witness.items():
                values = _closed(
                    raw["candidate_vs_control"],
                    {"output_max_abs", "gx_max_abs", "gw_max_abs"},
                    f"Liger runtime control_witness.{name}.candidate_vs_control",
                )
                for key, value in values.items():
                    _number(
                        value,
                        f"Liger runtime control_witness.{name}.candidate_vs_control.{key}",
                    )
                if values["gx_max_abs"] != 0.0:
                    raise TargetError(f"Liger runtime {name} control witness changed")
    except (ReplayError, KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        raise TargetError(f"Liger runtime witness is malformed: {exc}") from exc
    compiled = comparison["compiled"]
    compiled_expected = {
        "candidate_vs_reference",
        "control_vs_reference",
        "candidate_eager_vs_compiled",
        "control_eager_vs_compiled",
        "candidate_observation",
        "control_observation",
        "compile",
    }
    if not isinstance(compiled, dict) or set(compiled) != compiled_expected:
        raise TargetError("Liger target compiled runtime witness fields changed")
    for name in (
        "candidate_vs_reference",
        "control_vs_reference",
        "candidate_eager_vs_compiled",
        "control_eager_vs_compiled",
    ):
        try:
            values = compiled[name]
            if not isinstance(values, dict):
                raise TargetError(f"Liger runtime compiled.{name} is malformed")
            for key, value in _closed(
                values,
                {"output_max_abs", "gx_max_abs", "gw_max_abs"},
                f"Liger runtime compiled.{name}",
            ).items():
                _number(value, f"Liger runtime compiled.{name}.{key}")
        except (ReplayError, KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
            raise TargetError(f"Liger runtime witness is malformed: {exc}") from exc
    compile_evidence = compiled["compile"]
    if not isinstance(compile_evidence, dict) or set(compile_evidence) != {
        "candidate",
        "control",
    }:
        raise TargetError("Liger target compile witness lanes changed")
    _validate_compile_evidence(compile_evidence["candidate"])
    _validate_compile_evidence(compile_evidence["control"])
    _validate_runtime_evidence(comparison)


def _validate_eager_evidence(comparison):
    candidate_reference = comparison["candidate_vs_reference"]
    control_reference = comparison["control_vs_reference"]
    if (
        not math.isfinite(candidate_reference["gx_max_abs"])
        or not candidate_reference["gx_max_abs"] > 1e-4
    ):
        raise TargetError("in-place residual did not produce the expected numerical witness")
    if (
        not 0.0 <= candidate_reference["output_max_abs"] <= 2e-5
        or not 0.0 <= candidate_reference["gw_max_abs"] <= 2e-5
    ):
        raise TargetError("in-place Liger lane changed the reference output or dW")
    if any(
        not 0.0 <= control_reference[key] <= 2e-5
        for key in ("output_max_abs", "gx_max_abs", "gw_max_abs")
    ):
        raise TargetError("out-of-place Liger control does not match the pure-Torch reference")
    candidate_control = comparison["candidate_vs_control"]
    if candidate_control["output_max_abs"] != 0.0 or candidate_control["gw_max_abs"] != 0.0:
        raise TargetError("in-place and out-of-place lanes changed the forward output or dW")
    cotangent = comparison["cotangent"]
    if (
        not math.isfinite(cotangent["candidate_delta"])
        or not cotangent["candidate_delta"] > 1e-4
    ):
        raise TargetError("eager in-place lane did not mutate the external cotangent")
    if cotangent["control_delta"] != 0.0:
        raise TargetError("eager out-of-place lane mutated the external cotangent")
    if cotangent["candidate_version"] != [0, 0] or cotangent["control_version"] != [0, 0]:
        raise TargetError("eager Liger lane changed a cotangent tensor version")


def _compile_lane(function, x0, weight0, cotangent0):
    import torch
    try:
        from torch._inductor import metrics
    except (ImportError, OSError) as exc:
        raise TargetError("PyTorch Inductor is unavailable") from exc

    torch._dynamo.reset()
    metrics.reset()
    before = _counter_snapshot(torch)
    cold_start = time.perf_counter()
    compiled = torch.compile(function, fullgraph=True, dynamic=False, backend="inductor")
    lane = _run_lane(compiled, x0, weight0, cotangent0)
    cold_ms = (time.perf_counter() - cold_start) * 1000
    counters = _counter_delta(before, _counter_snapshot(torch))
    evidence = {
        "cold_ms": cold_ms,
        "generated_kernel_count": int(metrics.generated_kernel_count),
        "ir_nodes_pre_fusion": int(metrics.ir_nodes_pre_fusion),
        "counter_delta": counters,
    }
    _validate_compile_evidence(evidence)
    return compiled, lane, evidence


def _validate_compile_evidence(evidence):
    if not isinstance(evidence, dict) or set(evidence) != {
        "cold_ms",
        "generated_kernel_count",
        "ir_nodes_pre_fusion",
        "counter_delta",
    }:
        raise TargetError("Inductor lane compile evidence fields changed")
    cold_ms = evidence["cold_ms"]
    if (
        isinstance(cold_ms, bool)
        or not isinstance(cold_ms, (int, float))
        or not math.isfinite(cold_ms)
        or cold_ms <= 0
    ):
        raise TargetError("Inductor lane compile evidence has an invalid cold time")
    for name in ("generated_kernel_count", "ir_nodes_pre_fusion"):
        if not isinstance(evidence[name], int) or isinstance(evidence[name], bool):
            raise TargetError(f"Inductor lane compile evidence has an invalid {name}")
    counters = evidence["counter_delta"]
    if not isinstance(counters, dict) or any(
        not isinstance(values, dict)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values.values()
        )
        for values in counters.values()
    ):
        raise TargetError("Inductor lane compile evidence counters are malformed")
    aot = counters.get("aot_autograd", {})
    if aot.get("total", 0) != 1 or aot.get("ok", 0) != 1 or aot.get("not_ok", 0) != 0:
        raise TargetError(f"Inductor lane has an unexpected AOTAutograd counter delta: {aot}")
    inductor = counters.get("inductor", {})
    if inductor.get("extern_calls", 0) != 2:
        raise TargetError(
            f"Inductor lane expected 2 external calls, got {inductor.get('extern_calls', 0)}"
        )
    if evidence["generated_kernel_count"] != 4:
        raise TargetError(
            f"Inductor lane expected 4 generated kernels, got {evidence['generated_kernel_count']}"
        )
    if evidence["ir_nodes_pre_fusion"] != 9:
        raise TargetError(
            f"Inductor lane expected 9 pre-fusion IR nodes, got {evidence['ir_nodes_pre_fusion']}"
        )


def _control_evidence(function_class, x0, weight0, cotangent0):
    results: dict[str, Any] = {}
    target_witness: dict[str, Any] = {}
    for kind in ("isolated", "scaled"):
        candidate_fn = _liger_function(function_class, kind, True)
        control_fn = _liger_function(function_class, kind, False)
        candidate_capture = _capture(
            candidate_fn, x0, weight0, cotangent0, f"liger-{kind}-in-place"
        )
        control_capture = _capture(
            control_fn, x0, weight0, cotangent0, f"liger-{kind}-out-of-place"
        )
        result = _ownership.analyze(candidate_capture, control_capture)
        candidate_lane = _run_lane(candidate_fn, x0, weight0, cotangent0)
        control_lane = _run_lane(control_fn, x0, weight0, cotangent0)
        candidate_vs_control = _lane_delta(candidate_lane, control_lane)
        results[kind] = {"analysis": result.to_dict()}
        target_witness[kind] = {"candidate_vs_control": candidate_vs_control}
        if result.outcome.value != "VIOLATION" or result.collapse is not None:
            raise TargetError(
                f"{kind} control must show a borrowed write without contribution collapse"
            )
        if candidate_vs_control["gx_max_abs"] != 0.0:
            raise TargetError(f"{kind} control unexpectedly changed gx")
    safe_capture = _capture(
        _liger_function(function_class, "direct", False),
        x0,
        weight0,
        cotangent0,
        "liger-direct-out-of-place-control",
    )
    results["out_of_place"] = _ownership.analyze(safe_capture).to_dict()
    if results["out_of_place"]["outcome"] != "SAFE_IN_FRAGMENT":
        raise TargetError("out-of-place control was not accepted")
    return results, target_witness


def _liger_function(function_class, kind: str, in_place: bool):
    def function(x, weight):
        normalized = function_class.apply(x, weight, 1e-6, 1.0, "gemma", in_place, True)
        if kind == "direct":
            return x + normalized
        if kind == "isolated":
            return normalized
        if kind == "scaled":
            return normalized + 0.37 * x
        raise AssertionError(f"unreachable Liger fixture {kind}")

    return function


def _reference_function():
    import torch

    def function(x, weight):
        rstd = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6)
        normalized = x * rstd * (weight + 1.0)
        return x + normalized

    return function


def _run_lane(function, x0, weight0, cotangent0):
    import torch

    x = x0.clone().requires_grad_(True)
    weight = weight0.clone().requires_grad_(True)
    cotangent = cotangent0.clone()
    cotangent_before = cotangent.clone()
    version_before = cotangent._version
    torch.cuda.reset_peak_memory_stats()
    output = function(x, weight)
    gx, gw = torch.autograd.grad(output, (x, weight), cotangent)
    torch.cuda.synchronize()
    return {
        "output": output.detach().clone(),
        "gx": gx.detach().clone(),
        "gw": gw.detach().clone(),
        "cotangent_delta": float((cotangent - cotangent_before).abs().max().item()),
        "cotangent_version": [version_before, cotangent._version],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def _comparison(reference, candidate, control):
    return {
        "candidate_vs_reference": _lane_delta(candidate, reference),
        "control_vs_reference": _lane_delta(control, reference),
        "candidate_vs_control": _lane_delta(candidate, control),
        "cotangent": {
            "candidate_delta": candidate["cotangent_delta"],
            "control_delta": control["cotangent_delta"],
            "candidate_version": candidate["cotangent_version"],
            "control_version": control["cotangent_version"],
        },
        "peak_allocated_bytes": {
            "reference": reference["peak_allocated_bytes"],
            "candidate": candidate["peak_allocated_bytes"],
            "control": control["peak_allocated_bytes"],
        },
        "hashes": {
            "reference_gx": _tensor_hash(reference["gx"]),
            "candidate_gx": _tensor_hash(candidate["gx"]),
            "control_gx": _tensor_hash(control["gx"]),
        },
    }


def _lane_delta(left, right):
    return {
        "output_max_abs": _max_abs(left["output"], right["output"]),
        "gx_max_abs": _max_abs(left["gx"], right["gx"]),
        "gw_max_abs": _max_abs(left["gw"], right["gw"]),
    }


def _lane_observation(lane):
    return {
        "cotangent_delta": lane["cotangent_delta"],
        "cotangent_version": lane["cotangent_version"],
        "peak_allocated_bytes": lane["peak_allocated_bytes"],
    }


def _benchmark(functions, x0, weight0, cotangent0, *, warmups: int, samples: int):
    import torch

    lanes: dict[str, Any] = {}
    for name, function in functions.items():
        for _ in range(warmups):
            _run_lane(function, x0, weight0, cotangent0)
        device_samples: list[float] = []
        wall_samples: list[float] = []
        for _ in range(samples):
            x = x0.clone().requires_grad_(True)
            weight = weight0.clone().requires_grad_(True)
            cotangent = cotangent0.clone()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            start_event.record()
            output = function(x, weight)
            torch.autograd.grad(output, (x, weight), cotangent)
            end_event.record()
            torch.cuda.synchronize()
            wall_samples.append((time.perf_counter() - wall_start) * 1000)
            device_samples.append(float(start_event.elapsed_time(end_event)))
        lanes[name] = {
            "device_ms": device_samples,
            "wall_ms": wall_samples,
            "device_median_ms": statistics.median(device_samples),
            "wall_median_ms": statistics.median(wall_samples),
            "device_mad_ms": statistics.median(
                abs(value - statistics.median(device_samples)) for value in device_samples
            ),
        }
    return {
        "warmups": warmups,
        "samples": samples,
        "lanes": lanes,
        "claim": "supporting evidence only",
    }


def _load_liger_function(root: Path):
    _liger_packet._validate_liger_tree(root)
    for name in tuple(sys.modules):
        if name == "liger_kernel" or name.startswith("liger_kernel."):
            del sys.modules[name]
    package = types.ModuleType("liger_kernel")
    package.__path__ = []
    package.__package__ = "liger_kernel"
    ops_package = types.ModuleType("liger_kernel.ops")
    ops_package.__path__ = []
    ops_package.__package__ = "liger_kernel.ops"
    sys.modules["liger_kernel"] = package
    sys.modules["liger_kernel.ops"] = ops_package
    loaded = False
    try:
        _source_files._load_exact_module(
            "liger_kernel.utils",
            root / LIGER_ROOT_UTILS_PATH,
            LIGER_ROOT_UTILS_SHA256,
            "Liger",
        )
        _source_files._load_exact_module(
            "liger_kernel.ops.utils",
            root / LIGER_UTILS_PATH,
            LIGER_UTILS_SHA256,
            "Liger",
        )
        module = _source_files._load_exact_module(
            "liger_kernel.ops.rms_norm",
            root / LIGER_SOURCE_PATH,
            LIGER_SOURCE_SHA256,
            "Liger",
        )
        function_class = module.LigerRMSNormFunction
        loaded = True
        return function_class
    except (ImportError, OSError) as exc:
        raise TargetError(f"could not load pinned Liger modules: {exc}") from exc
    except AttributeError as exc:
        raise TargetError("pinned Liger module lacks LigerRMSNormFunction") from exc
    finally:
        if not loaded:
            for name in tuple(sys.modules):
                if name == "liger_kernel" or name.startswith("liger_kernel."):
                    del sys.modules[name]


def _target_identity(
    root: Path,
    candidate: Capture,
    packet: Path,
    *,
    control: Capture,
    runtime_witness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _liger_packet._validate_liger_tree(root)
    packet_prefix = Path("external") / "liger" / "tree"
    expected_kernels = {
        ("liger_kernel.ops.rms_norm", "_rms_norm_backward_kernel"),
        ("liger_kernel.ops.rms_norm", "_rms_norm_forward_kernel"),
    }
    bindings: dict[tuple[str, str, str], dict[str, str]] = {}
    for label, capture in (("candidate", candidate), ("control", control)):
        actual_kernels = {(kernel.module, kernel.qualname) for kernel in capture.kernels}
        if actual_kernels != expected_kernels or len(capture.kernels) != len(expected_kernels):
            raise TargetError(
                f"Liger {label} kernel set does not match the fixed row-mode specialization"
            )
        for kernel in capture.kernels:
            key = (kernel.module, kernel.qualname, kernel.source_sha256)
            bindings.setdefault(
                key,
                {
                    "module": kernel.module,
                    "qualname": kernel.qualname,
                    "file": (packet_prefix / LIGER_SOURCE_PATH).as_posix(),
                    "source_sha256": kernel.source_sha256,
                },
            )

    files = [
        _manifest_file(
            packet,
            packet_prefix / relative,
            relative,
            role,
        )
        for relative, _expected, role in _liger_packet._selected_liger_files()
    ]
    external_prefix = Path("external") / "liger"
    files.extend(
        (
            _manifest_file(
                packet,
                external_prefix / "NOTICE.json",
                "NOTICE.json",
                "notice",
            ),
            _manifest_file(
                packet,
                external_prefix / "UNSLOTH-APACHE-2.0.txt",
                "licenses/UNSLOTH-APACHE-2.0.txt",
                "license",
            ),
        )
    )
    specialization = {
        "shape": list(SHAPE),
        "dtype": "float32",
        "device": "cuda",
        "casting_mode": "gemma",
        "offset": 1.0,
        "epsilon": 1e-6,
        "elementwise_affine": True,
        "row_mode": True,
        "candidate_in_place": True,
        "control_in_place": False,
        "expression": "x + rms_norm(x)",
    }
    if runtime_witness is not None:
        _validate_liger_runtime_witness(runtime_witness)
        specialization["runtime_witness"] = runtime_witness
    return {
        "repository": LIGER_REPOSITORY,
        "revision": LIGER_REVISION,
        "files": files,
        "kernels": [bindings[key] for key in sorted(bindings)],
        "specialization": specialization,
    }


def _manifest_file(packet: Path, packet_path: Path, target_path: str, role: str) -> dict[str, Any]:
    data = _source_files._read_regular_file(packet / packet_path, packet_path, "Liger packet")
    return {
        "packet_path": packet_path.as_posix(),
        "target_path": target_path,
        "size": len(data),
        "sha256": sha256_bytes(data),
        "role": role,
    }
