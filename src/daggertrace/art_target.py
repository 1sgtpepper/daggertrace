from __future__ import annotations

import statistics
import threading
import time
from pathlib import Path
from typing import Any

from . import certificate as _certificate
from . import ownership as _ownership
from .errors import InvalidArtifact, TargetError, UnsupportedCapture
from .model import Analysis, Capture, Outcome
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
from .source_files import _read_verified_tree_file
from .art_capture import (
    _art_function,
    _reference_art_function,
    _capture_art,
    _load_art_module,
    _validate_art_audit_capture,
    _validate_art_runtime_bindings,
    _validate_art_capture_specialization,
    _observed_art_grids,
)
ART_REPOSITORY = "https://github.com/OpenPipe/ART"
ART_REVISION = "7585dda17553bd7996080fe645229b31011758cf"
ART_SOURCE_PATH = "src/art/trainer_rank/topk.py"
ART_SOURCE_SHA256 = "beda80e51c80db46bfc250bc1993237bee5281bad1b0deada79134f31cede2ed"
ART_LICENSE_PATH = "LICENSE"
ART_LICENSE_SHA256 = "d39185043a0ec089a0e5f88350f1f3eec10fbb34578e04aca3e2dab5b3a5f6cb"
ART_NOTICE_PATH = "THIRD-PARTY-NOTICES"
ART_NOTICE_SHA256 = "a778cc46e1a2a920d9fbdac4b93c669c71b205d34915a61d6b4f09f1e0453dd7"
ART_GPL_PATH = "licenses/GPL-3.0.txt"
ART_GPL_SHA256 = "3743f7a4ab5132f7dbeb88e68097657ab8374e74b46b6402efbe80ccfb1b6488"
ART_LGPL_PATH = "licenses/LGPL-3.0.txt"
ART_LGPL_SHA256 = "cfbd94ea9bfe6d0809fd2bebe2e0979cc8de9aca77f1a49eae11c474400e309e"
ART_TARGET_ID = "openpipe-art-local-stats"
ART_SHAPE = (2, 8192)
ART_K = 2
ART_BLOCK_V = 4096
ART_NUM_WARPS = 8
ART_SEED = 17

_ART_KERNEL_NAMES = (
    "_stats_stage1_kernel",
    "_stats_stage2_kernel",
    "_stats_backward_kernel",
)
# Direct reproductions have no capture from which to derive function bindings;
# these hashes are the slices of the separately pinned ART source file.
_ART_KERNEL_SOURCE_SHA256 = {
    "_stats_stage1_kernel": "5288b01552b260d7d4b83cdfbf0ce4965f2455b89f27795a252b527b9adabe95",
    "_stats_stage2_kernel": "4dcf22a2f0eb729503c6ef301ba67f5ce41d52e9965448a7bc38f3348bbf3850",
    "_stats_backward_kernel": "6af6295174b2ced7877a393f4ea1b6b4124fa5655310626fd210fcbaf5ea3fe6",
}
_ART_KERNEL_MODULE = "art.trainer_rank.topk"
_ART_CAPTURE_LOCK = threading.RLock()


def fetch_art(cache_dir: Path) -> Path:
    return _fetch_pinned_archive(
        cache_dir,
        root_name=f"ART-{ART_REVISION}",
        repository=ART_REPOSITORY,
        revision=ART_REVISION,
        user_agent="daggertrace/0.4",
        label="ART",
        validate_tree=_validate_art_tree,
    )


def audit_art(
    output_dir: Path,
    *,
    target_root: Path | None = None,
    warmups: int = 3,
    samples: int = 10,
) -> dict[str, Any]:
    """Run the candidate-only OpenPipe ART local-statistics demonstration."""

    _validate_benchmark_bounds(warmups, samples, "ART")
    torch = _require_module("torch", "PyTorch", "ART")
    triton = _require_module("triton", "Triton", "ART")
    if not torch.cuda.is_available():
        raise TargetError("the ART demonstration requires CUDA")
    _require_exact_gpu_framework(torch, triton, "the ART demonstration")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        root = target_root if target_root is not None else fetch_art(output_dir.parent / ".target-cache")
    except OSError as exc:
        raise TargetError(f"could not prepare the ART audit paths: {exc}") from exc

    module = _load_art_module(root)
    torch.manual_seed(ART_SEED)
    logits0 = torch.randn(ART_SHAPE, device="cuda", dtype=torch.float32)
    grad_output = torch.ones((ART_SHAPE[0],), device="cuda", dtype=torch.float32)
    function = _art_function(module)
    candidate, runtime_observations = _capture_art(function, logits0, grad_output)
    _validate_art_audit_capture(candidate, runtime_observations)
    analysis = _ownership.analyze(candidate)
    if analysis.outcome is not Outcome.SAFE_IN_FRAGMENT:
        raise TargetError(f"expected ART ownership result SAFE_IN_FRAGMENT: {analysis.to_dict()}")

    reference = _reference_art_function()
    runtime, target_runtime = _runtime_evidence_pair(function, logits0, grad_output)
    benchmark = _benchmark(
        {"torch_reference": reference, "art_local_stats": function},
        logits0,
        grad_output,
        warmups=warmups,
        samples=samples,
    )
    try:
        _stage_art_sources(root, output_dir)
        target = _target_identity(
            root,
            candidate,
            runtime_observations,
            runtime_witness=target_runtime,
        )
        _write_art_sidecars(output_dir, candidate, analysis, runtime, benchmark, target)
    except OSError as exc:
        raise TargetError(f"could not stage the ART audit artifact: {exc}") from exc
    return {
        "target_id": ART_TARGET_ID,
        "target": target,
        "candidate": candidate,
        "control": None,
        "analysis": analysis,
        "runtime": runtime,
        "target_runtime": target_runtime,
        "benchmark": benchmark,
        "controls": {},
        "artifacts": {
            "directory": output_dir.as_posix(),
            "certificate": (output_dir / "certificate.json").as_posix(),
            "candidate_capture": (output_dir / "candidate.capture.json").as_posix(),
            "control_capture": None,
            "reproducer": (output_dir / "reproducer.py").as_posix(),
        },
    }


def run_art_reproducer(target_root: Path | None = None) -> dict[str, Any]:
    """Run the minimal ART local-sum reproducer from an explicit source root, offline."""

    if target_root is None:
        raise TargetError(
            "the ART reproducer requires an explicit target_root and never fetches source"
        )

    torch = _require_module("torch", "PyTorch", "ART")
    triton = _require_module("triton", "Triton", "ART")
    if not torch.cuda.is_available():
        raise TargetError("the ART reproducer requires CUDA")
    _require_exact_gpu_framework(torch, triton, "the ART reproducer")
    root = target_root
    module = _load_art_module(root)
    torch.manual_seed(ART_SEED)
    logits0 = torch.randn(ART_SHAPE, device="cuda", dtype=torch.float32)
    grad_output = torch.ones((ART_SHAPE[0],), device="cuda", dtype=torch.float32)
    candidate = _run_art_lane(_art_function(module), logits0, grad_output)
    reference = _reference_art_lane(logits0, grad_output)
    comparison = {
        "target_id": ART_TARGET_ID,
        "candidate_vs_reference": _art_lane_delta(candidate, reference),
        "candidate": {"peak_allocated_bytes": candidate["peak_allocated_bytes"]},
        "reference": {"peak_allocated_bytes": reference["peak_allocated_bytes"]},
        "target": _target_identity(root, None),
    }
    _validate_art_runtime_evidence(comparison)
    return comparison


def _runtime_evidence_pair(function, logits0, grad_output) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_lane = _reference_art_lane(logits0, grad_output)
    candidate_lane = _run_art_lane(function, logits0, grad_output)
    compiled_lane, compile_evidence = _compile_art_lane(
        function,
        logits0,
        grad_output,
    )
    runtime = {
        "candidate_vs_reference": _art_lane_delta(candidate_lane, reference_lane),
        "control_vs_reference": None,
        "candidate_vs_control": None,
        "cotangent": {
            "candidate_delta": candidate_lane["cotangent_delta"],
            "control_delta": None,
            "candidate_version": candidate_lane["cotangent_version"],
            "control_version": None,
        },
        "peak_allocated_bytes": {"candidate": candidate_lane["peak_allocated_bytes"]},
        "hashes": {"candidate_gx": _tensor_hash(candidate_lane["gradient"])},
        "compiled": {
            "candidate_vs_reference": _art_lane_delta(compiled_lane, reference_lane),
            "control_vs_reference": None,
            "candidate_eager_vs_compiled": _art_lane_delta(candidate_lane, compiled_lane),
            "control_eager_vs_compiled": None,
            "candidate_observation": {
                "cotangent_delta": compiled_lane["cotangent_delta"],
                "cotangent_version": compiled_lane["cotangent_version"],
                "peak_allocated_bytes": compiled_lane["peak_allocated_bytes"],
            },
            "control_observation": None,
            "compile": {"candidate": compile_evidence},
        },
        "environment": _gpu_environment("ART"),
        "shape": list(ART_SHAPE),
        "seed": ART_SEED,
    }
    _validate_art_runtime_witness(runtime)
    return _generic_art_runtime(runtime), runtime


def _generic_art_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "environment": runtime["environment"],
        "candidate": {
            "compiled": {"occurrence": runtime["compiled"]["compile"]["candidate"]}
        },
        "control": None,
    }


def _compile_art_lane(function, logits0, grad_output):
    torch = _require_module("torch", "PyTorch", "ART")
    try:
        from torch._inductor import metrics
    except (ImportError, OSError) as exc:
        raise TargetError("PyTorch Inductor is unavailable") from exc

    torch._dynamo.reset()
    metrics.reset()
    before = _counter_snapshot(torch)
    start = time.perf_counter()
    compiled = torch.compile(function, fullgraph=True, dynamic=False, backend="inductor")
    lane = _run_art_lane(compiled, logits0, grad_output)
    cold_ms = (time.perf_counter() - start) * 1000
    evidence = {
        "cold_ms": cold_ms,
        "generated_kernel_count": int(metrics.generated_kernel_count),
        "ir_nodes_pre_fusion": int(metrics.ir_nodes_pre_fusion),
        "counter_delta": _counter_delta(before, _counter_snapshot(torch)),
    }
    _validate_art_compile_evidence(evidence)
    return lane, evidence


def _validate_art_compile_evidence(evidence: dict[str, Any]) -> None:
    counters = evidence["counter_delta"]
    aot = counters.get("aot_autograd", {})
    if aot.get("total") != 1 or aot.get("ok") != 1 or aot.get("not_ok", 0) != 0:
        raise TargetError(f"ART lane did not record one successful AOTAutograd occurrence: {aot}")
    inductor = counters.get("inductor", {})
    if inductor.get("extern_calls", 0) != 3:
        raise TargetError(f"ART lane expected 3 Inductor external calls: {inductor}")
    if evidence["generated_kernel_count"] != 1:
        raise TargetError("ART lane expected exactly 1 generated Inductor kernel")
    if evidence["ir_nodes_pre_fusion"] != 13:
        raise TargetError("ART lane expected exactly 13 pre-fusion IR nodes")


def _validate_art_runtime_evidence(comparison: dict[str, Any]) -> None:
    if not 0.0 <= comparison["candidate_vs_reference"]["output_max_abs"] <= 2e-4:
        raise TargetError("ART local_sum does not match the numerical reference")
    if not 0.0 <= comparison["candidate_vs_reference"]["gx_max_abs"] <= 2e-4:
        raise TargetError("ART local_sum gradient does not match the detached-max reference")
    compiled = comparison.get("compiled")
    if compiled is not None:
        if not 0.0 <= compiled["candidate_vs_reference"]["output_max_abs"] <= 2e-4:
            raise TargetError("compiled ART local_sum does not match the numerical reference")
        if not 0.0 <= compiled["candidate_vs_reference"]["gx_max_abs"] <= 2e-4:
            raise TargetError("compiled ART gradient does not match the numerical reference")
        if not 0.0 <= compiled["candidate_eager_vs_compiled"]["output_max_abs"] <= 2e-4:
            raise TargetError("compiled ART local_sum changed the eager result")
        if not 0.0 <= compiled["candidate_eager_vs_compiled"]["gx_max_abs"] <= 2e-4:
            raise TargetError("compiled ART gradient changed the eager result")


def _validate_art_runtime_witness(runtime: Any) -> None:
    """Validate the target-owned ART runtime record kept outside generic replay."""

    from .replay import ReplayError
    from .replay_packet import (
        _closed,
        _environment,
        _integer,
        _is_digest,
        _number,
        _runtime_lane,
    )

    try:
        evidence = _closed(
            runtime,
            {
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
            },
            "ART runtime witness",
        )
        _environment(evidence["environment"])

        def delta(raw: Any, label: str) -> None:
            value = _closed(
                raw,
                {"output_max_abs", "gx_max_abs", "gw_max_abs"},
                label,
            )
            for key in value:
                _number(value[key], f"{label}.{key}")

        def version(raw: Any, label: str) -> None:
            if not isinstance(raw, list) or len(raw) != 2:
                raise TargetError(f"{label} must contain two tensor versions")
            for index, value in enumerate(raw):
                _integer(value, f"{label}[{index}]", minimum=0)
            if raw[0] != raw[1]:
                raise TargetError(f"{label} changed the tensor version")

        delta(evidence["candidate_vs_reference"], "ART runtime candidate_vs_reference")
        if evidence["control_vs_reference"] is not None:
            raise TargetError("ART runtime candidate-only control comparison is not null")
        if evidence["candidate_vs_control"] is not None:
            raise TargetError("ART runtime candidate-only control lane is not null")

        cotangent = _closed(
            evidence["cotangent"],
            {"candidate_delta", "control_delta", "candidate_version", "control_version"},
            "ART runtime cotangent",
        )
        if _number(cotangent["candidate_delta"], "ART runtime cotangent.candidate_delta") != 0.0:
            raise TargetError("ART candidate lane mutated the external cotangent")
        if cotangent["control_delta"] is not None or cotangent["control_version"] is not None:
            raise TargetError("ART runtime candidate-only cotangent control is not null")
        version(cotangent["candidate_version"], "ART runtime cotangent.candidate_version")

        peak = _closed(
            evidence["peak_allocated_bytes"],
            {"candidate"},
            "ART runtime peak_allocated_bytes",
        )
        _integer(peak["candidate"], "ART runtime peak_allocated_bytes.candidate", minimum=0)
        hashes = _closed(evidence["hashes"], {"candidate_gx"}, "ART runtime hashes")
        if not _is_digest(hashes["candidate_gx"]):
            raise TargetError("ART runtime candidate gradient hash is malformed")

        compiled = _closed(
            evidence["compiled"],
            {
                "candidate_vs_reference",
                "control_vs_reference",
                "candidate_eager_vs_compiled",
                "control_eager_vs_compiled",
                "candidate_observation",
                "control_observation",
                "compile",
            },
            "ART runtime compiled",
        )
        delta(
            compiled["candidate_vs_reference"],
            "ART runtime compiled.candidate_vs_reference",
        )
        if compiled["control_vs_reference"] is not None:
            raise TargetError("ART compiled candidate-only control comparison is not null")
        delta(
            compiled["candidate_eager_vs_compiled"],
            "ART runtime compiled.candidate_eager_vs_compiled",
        )
        if compiled["control_eager_vs_compiled"] is not None:
            raise TargetError("ART compiled candidate-only control lane is not null")
        observation = _closed(
            compiled["candidate_observation"],
            {"cotangent_delta", "cotangent_version", "peak_allocated_bytes"},
            "ART runtime compiled.candidate_observation",
        )
        if _number(
            observation["cotangent_delta"],
            "ART runtime compiled.candidate_observation.cotangent_delta",
        ) != 0.0:
            raise TargetError("compiled ART lane mutated the external cotangent")
        version(
            observation["cotangent_version"],
            "ART runtime compiled.candidate_observation.cotangent_version",
        )
        _integer(
            observation["peak_allocated_bytes"],
            "ART runtime compiled.candidate_observation.peak_allocated_bytes",
            minimum=0,
        )
        if compiled["control_observation"] is not None:
            raise TargetError("ART compiled candidate-only control observation is not null")
        compile = _closed(compiled["compile"], {"candidate"}, "ART runtime compiled.compile")
        _runtime_lane(
            {"compiled": {"occurrence": compile["candidate"]}},
            "ART runtime compiled.compile.candidate",
            required=True,
        )
        _validate_art_compile_evidence(compile["candidate"])

        if evidence["shape"] != list(ART_SHAPE):
            raise TargetError("ART runtime shape changed")
        if _integer(evidence["seed"], "ART runtime seed", minimum=0) != ART_SEED:
            raise TargetError("ART runtime seed changed")
        _validate_art_runtime_evidence(evidence)
    except TargetError:
        raise
    except (
        ReplayError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        IndexError,
        OverflowError,
    ) as exc:
        raise TargetError(f"ART runtime witness is malformed: {exc}") from exc


def _run_art_lane(function, logits0, grad_output) -> dict[str, Any]:
    torch = _require_module("torch", "PyTorch", "ART")
    logits = logits0.clone().requires_grad_(True)
    cotangent_before = grad_output.detach().clone()
    cotangent_version = grad_output._version
    torch.cuda.reset_peak_memory_stats()
    local_sum = function(logits)
    gradient = torch.autograd.grad(local_sum, logits, grad_outputs=grad_output)
    torch.cuda.synchronize()
    return {
        "local_sum": local_sum.detach().clone(),
        "gradient": gradient[0].detach().clone(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cotangent_delta": _max_abs(grad_output, cotangent_before),
        "cotangent_version": [cotangent_version, grad_output._version],
    }


def _reference_art_lane(logits0, grad_output) -> dict[str, Any]:
    torch = _require_module("torch", "PyTorch", "ART")
    logits = logits0.detach()
    row_max = logits.amax(dim=-1, keepdim=True).detach()
    local_sum = torch.exp(logits - row_max).sum(dim=-1)
    gradient = grad_output.unsqueeze(-1) * torch.exp(logits - row_max)
    return {
        "local_sum": local_sum.detach().clone(),
        "gradient": gradient.detach().clone(),
        "peak_allocated_bytes": 0,
    }


def _art_lane_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        "output_max_abs": _max_abs(left["local_sum"], right["local_sum"]),
        "gx_max_abs": _max_abs(left["gradient"], right["gradient"]),
        "gw_max_abs": 0.0,
    }


def _benchmark(functions, logits0, grad_output, *, warmups: int, samples: int) -> dict[str, Any]:
    torch = _require_module("torch", "PyTorch", "ART")
    lanes: dict[str, Any] = {}
    for name, function in functions.items():
        for _ in range(warmups):
            _run_art_lane(function, logits0, grad_output)
        device_samples: list[float] = []
        wall_samples: list[float] = []
        for _ in range(samples):
            logits = logits0.clone().requires_grad_(True)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            start_event.record()
            output = function(logits)
            torch.autograd.grad(output, logits, grad_outputs=grad_output)
            end_event.record()
            torch.cuda.synchronize()
            wall_samples.append((time.perf_counter() - wall_start) * 1000)
            device_samples.append(float(start_event.elapsed_time(end_event)))
        median = statistics.median(device_samples)
        lanes[name] = {
            "device_ms": device_samples,
            "wall_ms": wall_samples,
            "device_median_ms": median,
            "wall_median_ms": statistics.median(wall_samples),
            "device_mad_ms": statistics.median(abs(value - median) for value in device_samples),
        }
    return {"warmups": warmups, "samples": samples, "lanes": lanes, "claim": "supporting evidence only"}


def _stage_art_sources(root: Path, output_dir: Path) -> None:
    tree = output_dir / "external" / "openpipe-art" / "tree"
    for relative, expected, _role in _selected_art_files():
        destination = tree / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_read_verified_tree_file(root, relative, expected, "ART"))


def _target_identity(
    root: Path,
    candidate: Capture | None,
    runtime_observations: list[dict[str, Any]] | None = None,
    runtime_witness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Reproducer output uses the same packet-root-relative paths as certificates,
    # even when it is run directly against an external checkout.
    packet_prefix = Path("external") / "openpipe-art" / "tree"
    bindings = [
        {
            "module": _ART_KERNEL_MODULE,
            "qualname": qualname,
            "file": (packet_prefix / ART_SOURCE_PATH).as_posix(),
            "source_sha256": _ART_KERNEL_SOURCE_SHA256[qualname],
        }
        for qualname in sorted(_ART_KERNEL_NAMES)
    ]
    if candidate is not None:
        bindings = [
            {
                "module": kernel.module,
                "qualname": kernel.qualname,
                "file": (packet_prefix / ART_SOURCE_PATH).as_posix(),
                "source_sha256": kernel.source_sha256,
            }
            for kernel in sorted(candidate.kernels, key=lambda item: (item.module, item.qualname))
        ]
    files = []
    for relative, expected, role in _selected_art_files():
        data = _read_verified_tree_file(root, relative, expected, "ART")
        files.append(
            {
                "packet_path": (packet_prefix / relative).as_posix(),
                "target_path": relative,
                "size": len(data),
                "sha256": expected,
                "role": role,
            }
        )
    specialization = {
        "shape": list(ART_SHAPE),
        "dtype": "float32",
        "device": "cuda",
        "k": ART_K,
        "block_v": ART_BLOCK_V,
        "expression": "local_sum",
        "row_max": "detached",
        "source_launches": {
            "decorator": "triton.jit",
            "num_warps": ART_NUM_WARPS,
            "kernel_count": len(_ART_KERNEL_NAMES),
        },
    }
    if candidate is not None:
        if not isinstance(runtime_observations, list):
            raise TargetError("ART target identity requires observed HOP runtime metadata")
        observations = runtime_observations
        specialization["runtime_capture"] = {
            "kernel_multiplicity": len(observations),
            "wrapper_chain": "triton.runtime.autotuner.Autotuner -> "
            "triton.runtime.jit.JITFunction",
            "observations": observations,
            "grids": _observed_art_grids(candidate),
        }
    else:
        specialization["runtime_capture"] = None
    if runtime_witness is not None:
        _validate_art_runtime_witness(runtime_witness)
        specialization["runtime_witness"] = runtime_witness
    return {
        "repository": ART_REPOSITORY,
        "revision": ART_REVISION,
        "files": files,
        "kernels": bindings,
        "specialization": specialization,
    }


def _selected_art_files() -> tuple[tuple[str, str, str], ...]:
    return (
        (ART_SOURCE_PATH, ART_SOURCE_SHA256, "source"),
        (ART_LICENSE_PATH, ART_LICENSE_SHA256, "license"),
        (ART_NOTICE_PATH, ART_NOTICE_SHA256, "notice"),
        (ART_GPL_PATH, ART_GPL_SHA256, "license"),
        (ART_LGPL_PATH, ART_LGPL_SHA256, "license"),
    )


def _write_art_sidecars(
    output_dir: Path,
    candidate: Capture,
    analysis: Analysis,
    runtime: dict[str, Any],
    benchmark: dict[str, Any],
    target: dict[str, Any],
) -> None:
    if set(target) != {"repository", "revision", "files", "kernels", "specialization"}:
        raise InvalidArtifact("ART target identity must use the closed generic target fields")
    specialization = target["specialization"]
    if (
        not isinstance(specialization, dict)
        or not isinstance(specialization.get("runtime_witness"), dict)
    ):
        raise InvalidArtifact("ART target identity must carry its runtime witness")
    controls = {}
    validation = {
        "oracle_cases": 0,
        "oracle_passed": 0,
        "mutation_cases": 0,
        "mutations_killed": 0,
        "ablations": 0,
        "ablations_observed": 0,
    }
    certificate = _certificate.make_certificate(
        candidate=candidate,
        control=None,
        analysis=analysis,
        target=target,
        runtime=runtime,
        benchmark=benchmark,
        controls=controls,
        validation=validation,
    )
    _certificate.write_certificate(
        output_dir / "certificate.json",
        certificate,
        reproducer_bytes=_art_reproducer_script().encode("utf-8"),
    )


def _art_reproducer_script() -> str:
    return """import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from daggertrace.art_target import run_art_reproducer

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "external" / "openpipe-art" / "tree"
    print(json.dumps(run_art_reproducer(target), sort_keys=True))
"""


def _replay_art_loaded(
    certificate: dict[str, Any],
    certificate_path: Path,
    generic_result: dict[str, Any],
) -> dict[str, Any]:
    """Validate ART-specific data in the certificate loaded by generic replay."""

    try:
        _require_generic_target_result(
            generic_result,
            outcome="SAFE_IN_FRAGMENT",
            reasons=["FRESH_DESTINATION"],
            borrowed_writes=0,
            contextual_collapse=False,
            target="ART",
        )
        packet = certificate_path.parent.resolve()
        if certificate_path.name != "certificate.json":
            raise TargetError("ART packet certificate must be named certificate.json")
        target = certificate["target"]
        if (
            not isinstance(target, dict)
            or target.get("repository") != ART_REPOSITORY
            or target.get("revision") != ART_REVISION
        ):
            raise TargetError("ART target repository or revision changed")
        specialization = target.get("specialization")
        if not isinstance(specialization, dict):
            raise TargetError("ART target specialization changed")
        runtime_capture = specialization.get("runtime_capture")
        if not isinstance(runtime_capture, dict):
            raise TargetError("ART runtime capture identity is missing")
        observations = runtime_capture.get("observations")
        if not isinstance(observations, list):
            raise TargetError("ART runtime observations are missing")
        runtime_witness = specialization.get("runtime_witness")
        if not isinstance(runtime_witness, dict):
            raise TargetError("ART target runtime witness is missing")
        _validate_art_runtime_witness(runtime_witness)
        if certificate["runtime"] != _generic_art_runtime(runtime_witness):
            raise TargetError("ART generic runtime lane does not match its target witness")
        candidate = Capture.from_dict(certificate["candidate"])
        _validate_art_runtime_bindings(candidate, observations)
        tree = packet / "external" / "openpipe-art" / "tree"
        if target != _target_identity(
            tree,
            candidate,
            observations,
            runtime_witness=runtime_witness,
        ):
            raise TargetError("ART target identity does not match supplied source/capture bytes")
        _validate_art_capture_specialization(candidate)
        return {
            "adapter": ART_TARGET_ID,
            "adapter_only": False,
            "generic_replay_compatible": True,
            "replay": "generic-then-art-target-closure",
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
        raise TargetError(f"ART packet could not be replayed: {exc}") from exc


def _validate_art_tree(root: Path) -> None:
    for relative, expected, _role in _selected_art_files():
        _read_verified_tree_file(root, relative, expected, "ART")
