"""Collect a blind, target-adapter-free AOT/Triton holdout result.

The three holdout modules are loaded from this directory after the checker source
digest is recorded. They are not imported by ``src/daggertrace`` and use no target
adapter. Target-adapter-free describes the checker boundary; executing the holdout
still requires the pinned GPU/Triton environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
HOLDOUT = Path(__file__).with_name("holdout") / "holdout_kernel.py"
AFFINE_HOLDOUT = Path(__file__).with_name("holdout") / "affine_holdout.py"
REDUCTION_HOLDOUT = Path(__file__).with_name("holdout") / "reduction_holdout.py"
MAX_HOLDOUT_SIZE = 8192
TARGET_ADAPTER_MODULES = {
    "daggertrace.targets",
    "daggertrace.art_capture",
    "daggertrace.art_target",
    "daggertrace.liger_capture",
    "daggertrace.liger_packet",
    "daggertrace.liger_target",
    "daggertrace.memory_model_target",
}
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from daggertrace.canonical import sha256_file, sha256_json, write_canonical_json  # noqa: E402
from daggertrace.capture import capture_aot  # noqa: E402
from daggertrace.ownership import analyze  # noqa: E402


def _source_state() -> dict[str, Any]:
    files = {
        path.relative_to(REPO).as_posix(): sha256_file(path)
        for path in sorted((REPO / "src" / "daggertrace").rglob("*.py"))
    }
    if (REPO / ".git").exists():
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "src/daggertrace"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        source_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source_tree_dirty = bool(status.strip())
    else:
        marker = REPO / "SOURCE_REVISION"
        try:
            source_revision = marker.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(
                "holdout provenance requires Git metadata or a SOURCE_REVISION file"
            ) from exc
        if len(source_revision) != 40 or any(
            character not in "0123456789abcdef" for character in source_revision
        ):
            raise RuntimeError("SOURCE_REVISION must contain a lowercase 40-hex revision")
        source_tree_dirty = None
    return {
        "commit": source_revision,
        "source_manifest": files,
        "source_manifest_sha256": sha256_json(files),
        "source_tree_dirty": source_tree_dirty,
    }


_PROOF_BOOLEAN_KEYS = frozenset(
    {
        "all_numeric_checks_passed",
        "baseline_safe_cases_passed",
        "composed_control_safe",
        "composed_candidate_violation_detected",
        "composed_contribution_collapse_detected",
        "multi_input_saved_state_cases_passed",
        "compiled_affine_case_passed",
        "reduction_cases_passed",
        "compiled_reduction_cases_passed",
    }
)


def _proof_valid(conclusion: Any) -> bool:
    if not isinstance(conclusion, dict):
        return False
    if set(conclusion) != _PROOF_BOOLEAN_KEYS | {"generalization_scope"}:
        return False
    scope = conclusion["generalization_scope"]
    return (
        all(conclusion[key] is True for key in _PROOF_BOOLEAN_KEYS)
        and isinstance(scope, str)
        and bool(scope)
    )


def _load_module(path: Path, name: str):
    if path.is_relative_to(REPO / "src"):
        raise RuntimeError("holdout must remain outside the checker source tree")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load holdout module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_holdout_sources_unchanged(digests: dict[Path, str]) -> None:
    """Ensure recorded holdout digests still describe the loaded sources."""

    for path, digest in digests.items():
        try:
            unchanged = sha256_file(path) == digest
        except OSError as exc:
            raise RuntimeError(f"holdout source changed during collection: {path.name}") from exc
        if not unchanged:
            raise RuntimeError(f"holdout source changed during collection: {path.name}")


def _max_abs(left, right) -> float:
    value = float((left - right).abs().max().item())
    if not math.isfinite(value):
        raise RuntimeError("holdout numeric result was non-finite")
    return value


def _assert_finite(value: Any, label: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"{label} contains a non-finite number")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite(child, f"{label}[{index}]")


def _kernel_record(kernel: Any) -> dict[str, Any]:
    runtime = kernel.runtime
    return {
        "id": kernel.id,
        "module": kernel.module,
        "qualname": kernel.qualname,
        "source_sha256": kernel.source_sha256,
        "reads": list(kernel.reads),
        "writes": list(kernel.writes),
        "runtime": runtime,
    }


def _run_shape(module: Any, size: int) -> dict[str, Any]:
    import torch

    torch.manual_seed(1000 + size)
    x = torch.randn(size, device="cuda", dtype=torch.float32, requires_grad=True)
    output = module.apply(x)
    expected_output = x.detach() + 1.0
    dy = torch.randn_like(output)
    gradient = torch.autograd.grad(output, x, dy)[0]
    torch.cuda.synchronize()

    capture_x = torch.randn(size, device="cuda", dtype=torch.float32, requires_grad=True)
    capture_dy = torch.randn_like(capture_x)
    capture = capture_aot(module.apply, (capture_x,), capture_dy, label=f"holdout-{size}")
    result = analyze(capture).to_dict()
    if result["outcome"] != "SAFE_IN_FRAGMENT" or result["reasons"] != ["FRESH_DESTINATION"]:
        raise RuntimeError(f"holdout ownership result changed: {result}")
    if result["borrowed_writes"] or result["collapse"] is not None:
        raise RuntimeError(f"holdout unexpectedly borrowed or collapsed a contribution: {result}")
    if len(capture.kernels) != 2:
        raise RuntimeError(f"holdout expected two kernels, got {len(capture.kernels)}")
    return {
        "size": size,
        "output_max_abs": _max_abs(output.detach(), expected_output),
        "gradient_max_abs": _max_abs(gradient, dy),
        "capture_digest": capture.digest(),
        "capture": capture.to_dict(),
        "analysis": result,
        "kernels": [_kernel_record(kernel) for kernel in capture.kernels],
        "aot_callbacks": capture.diagnostics["callbacks"],
        "aot_autograd_delta": capture.diagnostics["aot_autograd"],
    }


def _run_composed_shape(module: Any, size: int) -> dict[str, Any]:
    import torch

    torch.manual_seed(2000 + size)
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    dy = torch.randn_like(x)
    expected_output = x * 2.5 + 0.25
    expected_gradient = dy * 2.5

    lanes: dict[str, dict[str, Any]] = {}
    for name, function in (
        ("safe", module.composed_safe),
        ("unsafe", module.composed_unsafe),
    ):
        lane_x = x.detach().clone().requires_grad_(True)
        lane_dy = dy.detach().clone()
        output = function(lane_x)
        gradient = torch.autograd.grad(output, lane_x, lane_dy)[0]
        lanes[name] = {
            "output_max_abs": _max_abs(output.detach(), expected_output),
            "gradient_vs_reference_max_abs": _max_abs(gradient, expected_gradient),
            "gradient": gradient.detach(),
            "cotangent_delta_max_abs": _max_abs(lane_dy, dy),
        }

    captures: dict[str, Any] = {}
    analyses: dict[str, dict[str, Any]] = {}
    for name, function in (
        ("safe", module.composed_safe),
        ("unsafe", module.composed_unsafe),
    ):
        capture_x = torch.randn(size, device="cuda", dtype=torch.float32, requires_grad=True)
        capture_dy = torch.randn_like(capture_x)
        capture = capture_aot(
            function,
            (capture_x,),
            capture_dy,
            label=f"holdout-composed-{name}-{size}",
        )
        captures[name] = capture
        analyses[name] = analyze(capture).to_dict()

    pair_analysis = analyze(captures["unsafe"], captures["safe"]).to_dict()
    if (
        analyses["safe"]["outcome"] != "SAFE_IN_FRAGMENT"
        or analyses["safe"]["reasons"] != ["FRESH_DESTINATION"]
    ):
        raise RuntimeError(f"composed safe holdout classification changed: {analyses['safe']}")
    if analyses["unsafe"]["outcome"] != "VIOLATION" or analyses["unsafe"]["reasons"] != [
        "BORROWED_COTANGENT_WRITE"
    ]:
        raise RuntimeError(f"composed unsafe holdout classification changed: {analyses['unsafe']}")
    if pair_analysis["outcome"] != "VIOLATION" or pair_analysis["reasons"] != [
        "BORROWED_COTANGENT_WRITE",
        "CONTRIBUTION_COLLAPSE",
    ]:
        raise RuntimeError(f"composed pair classification changed: {pair_analysis}")
    if any(len(capture.kernels) != 2 for capture in captures.values()):
        raise RuntimeError("composed holdout expected two kernels per lane")

    return {
        "size": size,
        "numeric": {
            "safe": {
                key: value
                for key, value in lanes["safe"].items()
                if key != "gradient"
            },
            "unsafe": {
                key: value
                for key, value in lanes["unsafe"].items()
                if key != "gradient"
            },
            "unsafe_vs_safe_gradient_max_abs": _max_abs(
                lanes["unsafe"]["gradient"], lanes["safe"]["gradient"]
            ),
        },
        "captures": {
            name: {
                "digest": capture.digest(),
                "capture": capture.to_dict(),
                "analysis": analyses[name],
                "kernels": [_kernel_record(kernel) for kernel in capture.kernels],
                "aot_callbacks": capture.diagnostics["callbacks"],
                "aot_autograd_delta": capture.diagnostics["aot_autograd"],
            }
            for name, capture in captures.items()
        },
        "paired_analysis": pair_analysis,
    }


def _run_affine_shape(module: Any, size: int) -> dict[str, Any]:
    import torch

    torch.manual_seed(3000 + size)
    x = torch.randn(size, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(size, device="cuda", dtype=torch.float32, requires_grad=True)
    dy = torch.randn_like(x)
    output = module.apply(x, weight)
    gradient_x, gradient_weight = torch.autograd.grad(output, (x, weight), dy)
    expected_output = x.detach() * weight.detach() + 0.25
    expected_x = dy * weight.detach()
    expected_weight = dy * x.detach()

    capture_x = torch.randn(size, device="cuda", dtype=torch.float32, requires_grad=True)
    capture_weight = torch.randn(size, device="cuda", dtype=torch.float32, requires_grad=True)
    capture_dy = torch.randn_like(capture_x)
    capture = capture_aot(
        module.apply,
        (capture_x, capture_weight),
        capture_dy,
        label=f"holdout-affine-{size}",
    )
    analysis = analyze(capture).to_dict()
    if analysis["outcome"] != "SAFE_IN_FRAGMENT" or analysis["reasons"] != ["FRESH_DESTINATION"]:
        raise RuntimeError(f"affine holdout ownership result changed: {analysis}")
    if analysis["borrowed_writes"] or analysis["collapse"] is not None:
        raise RuntimeError(f"affine holdout unexpectedly borrowed or collapsed: {analysis}")
    if len(capture.kernels) != 2 or len(capture.backward.output_roles) != 2:
        raise RuntimeError("affine holdout expected two kernels and two gradient slots")

    torch._dynamo.reset()
    torch._inductor.metrics.reset()
    counter_before = dict(torch._dynamo.utils.counters["aot_autograd"])
    compiled_function = torch.compile(
        module.apply,
        fullgraph=True,
        dynamic=False,
        backend="inductor",
    )
    compiled_x = x.detach().clone().requires_grad_(True)
    compiled_weight = weight.detach().clone().requires_grad_(True)
    compiled_dy = dy.detach().clone()
    compiled_output = compiled_function(compiled_x, compiled_weight)
    compiled_x_grad, compiled_weight_grad = torch.autograd.grad(
        compiled_output,
        (compiled_x, compiled_weight),
        compiled_dy,
    )
    torch.cuda.synchronize()
    counter_after = dict(torch._dynamo.utils.counters["aot_autograd"])
    compiled = {
        "size": size,
        "backend": "inductor",
        "output_max_abs": _max_abs(compiled_output, output.detach()),
        "gradient_x_max_abs": _max_abs(compiled_x_grad, gradient_x),
        "gradient_weight_max_abs": _max_abs(compiled_weight_grad, gradient_weight),
        "aot_autograd_delta": {
            key: counter_after.get(key, 0) - counter_before.get(key, 0)
            for key in sorted(set(counter_before) | set(counter_after))
            if counter_after.get(key, 0) != counter_before.get(key, 0)
        },
        "inductor_metrics": {
            "generated_kernel_count": int(torch._inductor.metrics.generated_kernel_count),
            "ir_nodes_pre_fusion": int(torch._inductor.metrics.ir_nodes_pre_fusion),
        },
    }
    if any(
        compiled[key] > 1e-6
        for key in ("output_max_abs", "gradient_x_max_abs", "gradient_weight_max_abs")
    ):
        raise RuntimeError(f"affine Inductor lane diverged from eager: {compiled}")
    if compiled["aot_autograd_delta"].get("total") != 1 or compiled["aot_autograd_delta"].get("ok") != 1:
        raise RuntimeError(f"affine Inductor AOT occurrence changed: {compiled}")

    return {
        "size": size,
        "numeric": {
            "output_max_abs": _max_abs(output.detach(), expected_output),
            "gradient_x_max_abs": _max_abs(gradient_x, expected_x),
            "gradient_weight_max_abs": _max_abs(gradient_weight, expected_weight),
        },
        "capture_digest": capture.digest(),
        "capture": capture.to_dict(),
        "analysis": analysis,
        "kernels": [_kernel_record(kernel) for kernel in capture.kernels],
        "aot_callbacks": capture.diagnostics["callbacks"],
        "aot_autograd_delta": capture.diagnostics["aot_autograd"],
        "compiled": compiled,
    }


def _run_reduction_shape(module: Any, size: int) -> dict[str, Any]:
    import torch

    torch.manual_seed(4000 + size)
    x = torch.randn(size, device="cuda", dtype=torch.float32, requires_grad=True)
    dy = torch.ones(1, device="cuda", dtype=torch.float32)
    output = module.apply(x)
    gradient = torch.autograd.grad(output, x, dy)[0]
    expected_output = x.detach().sum().reshape(1)
    expected_gradient = torch.ones_like(x)

    capture_x = torch.randn(size, device="cuda", dtype=torch.float32, requires_grad=True)
    capture_dy = torch.ones(1, device="cuda", dtype=torch.float32)
    capture = capture_aot(
        module.apply,
        (capture_x,),
        capture_dy,
        label=f"holdout-reduction-{size}",
    )
    analysis = analyze(capture).to_dict()
    if analysis["outcome"] != "SAFE_IN_FRAGMENT" or analysis["reasons"] != ["FRESH_DESTINATION"]:
        raise RuntimeError(f"reduction holdout ownership result changed: {analysis}")
    if analysis["borrowed_writes"] or analysis["collapse"] is not None:
        raise RuntimeError(f"reduction holdout unexpectedly borrowed or collapsed: {analysis}")
    if len(capture.kernels) != 2 or len(capture.backward.output_roles) != 1:
        raise RuntimeError("reduction holdout expected two kernels and one gradient slot")

    torch._dynamo.reset()
    torch._inductor.metrics.reset()
    counter_before = dict(torch._dynamo.utils.counters["aot_autograd"])
    compiled_function = torch.compile(
        module.apply,
        fullgraph=True,
        dynamic=False,
        backend="inductor",
    )
    compiled_x = x.detach().clone().requires_grad_(True)
    compiled_output = compiled_function(compiled_x)
    compiled_gradient = torch.autograd.grad(compiled_output, compiled_x, dy)[0]
    torch.cuda.synchronize()
    counter_after = dict(torch._dynamo.utils.counters["aot_autograd"])
    compiled = {
        "size": size,
        "backend": "inductor",
        "output_max_abs": _max_abs(compiled_output, output.detach()),
        "gradient_max_abs": _max_abs(compiled_gradient, gradient),
        "aot_autograd_delta": {
            key: counter_after.get(key, 0) - counter_before.get(key, 0)
            for key in sorted(set(counter_before) | set(counter_after))
            if counter_after.get(key, 0) != counter_before.get(key, 0)
        },
        "inductor_metrics": {
            "generated_kernel_count": int(torch._inductor.metrics.generated_kernel_count),
            "ir_nodes_pre_fusion": int(torch._inductor.metrics.ir_nodes_pre_fusion),
        },
    }
    if compiled["output_max_abs"] > 1e-6 or compiled["gradient_max_abs"] > 1e-6:
        raise RuntimeError(f"reduction Inductor lane diverged from eager: {compiled}")
    if compiled["aot_autograd_delta"].get("total") != 1 or compiled["aot_autograd_delta"].get("ok") != 1:
        raise RuntimeError(f"reduction Inductor AOT occurrence changed: {compiled}")

    return {
        "size": size,
        "numeric": {
            "output_max_abs": _max_abs(output.detach(), expected_output),
            "gradient_max_abs": _max_abs(gradient, expected_gradient),
        },
        "capture_digest": capture.digest(),
        "capture": capture.to_dict(),
        "analysis": analysis,
        "kernels": [_kernel_record(kernel) for kernel in capture.kernels],
        "aot_callbacks": capture.diagnostics["callbacks"],
        "aot_autograd_delta": capture.diagnostics["aot_autograd"],
        "compiled": compiled,
    }


def collect(sizes: tuple[int, ...]) -> dict[str, Any]:
    try:
        sizes = tuple(sizes)
    except TypeError as exc:
        raise ValueError("sizes must be an iterable of integers") from exc
    if not sizes:
        raise ValueError("at least one holdout size is required")
    if any(type(size) is not int or not 1 <= size <= MAX_HOLDOUT_SIZE for size in sizes):
        raise ValueError(
            f"holdout sizes must be unique integers between 1 and {MAX_HOLDOUT_SIZE}"
        )
    if len(set(sizes)) != len(sizes):
        raise ValueError("holdout sizes must not contain duplicates")

    imported_adapters = sorted(name for name in TARGET_ADAPTER_MODULES if name in sys.modules)
    if imported_adapters:
        raise RuntimeError(f"blind holdout imported target adapters: {imported_adapters}")

    before = _source_state()
    holdout_digests = {
        path: sha256_file(path)
        for path in (HOLDOUT, AFFINE_HOLDOUT, REDUCTION_HOLDOUT)
    }
    module = _load_module(HOLDOUT, "local_holdout_kernel")
    affine_module = _load_module(AFFINE_HOLDOUT, "local_affine_holdout")
    reduction_module = _load_module(REDUCTION_HOLDOUT, "local_reduction_holdout")
    _assert_holdout_sources_unchanged(holdout_digests)
    holdout_sha256 = holdout_digests[HOLDOUT]
    affine_holdout_sha256 = holdout_digests[AFFINE_HOLDOUT]
    reduction_holdout_sha256 = holdout_digests[REDUCTION_HOLDOUT]
    results = [_run_shape(module, size) for size in sizes]
    composed_results = [_run_composed_shape(module, size) for size in sizes]
    affine_results = [_run_affine_shape(affine_module, size) for size in sizes]
    reduction_results = [_run_reduction_shape(reduction_module, size) for size in sizes]
    imported_adapters = sorted(name for name in TARGET_ADAPTER_MODULES if name in sys.modules)
    if imported_adapters:
        raise RuntimeError(f"blind holdout imported target adapters: {imported_adapters}")
    _assert_holdout_sources_unchanged(holdout_digests)
    expected_sizes = set(sizes)
    for family_name, family in (
        ("baseline", results),
        ("composed", composed_results),
        ("affine", affine_results),
        ("reduction", reduction_results),
    ):
        observed_sizes = [
            row.get("size")
            for row in family
            if isinstance(row, dict) and type(row.get("size")) is int
        ]
        if (
            len(observed_sizes) != len(family)
            or len(family) != len(sizes)
            or set(observed_sizes) != expected_sizes
        ):
            raise RuntimeError(f"{family_name} holdout family coverage is incomplete")
        _assert_finite(family, f"{family_name} holdout family")
    after = _source_state()
    if before != after:
        raise RuntimeError("checker source changed during holdout collection")
    if any(result["output_max_abs"] != 0.0 for result in results):
        raise RuntimeError("holdout output did not match its reference")
    if any(result["gradient_max_abs"] != 0.0 for result in results):
        raise RuntimeError("holdout gradient did not match its reference")
    for result in composed_results:
        safe = result["numeric"]["safe"]
        unsafe = result["numeric"]["unsafe"]
        if safe["output_max_abs"] > 2e-5 or safe["gradient_vs_reference_max_abs"] > 2e-5:
            raise RuntimeError(f"composed safe lane did not match its reference: {result}")
        if unsafe["output_max_abs"] > 2e-5:
            raise RuntimeError(f"composed unsafe output changed: {result}")
        if unsafe["gradient_vs_reference_max_abs"] <= 1e-4:
            raise RuntimeError(f"composed unsafe lane lost its gradient witness: {result}")
        if result["numeric"]["unsafe_vs_safe_gradient_max_abs"] <= 1e-4:
            raise RuntimeError(f"composed lanes became numerically indistinguishable: {result}")
        if safe["cotangent_delta_max_abs"] > 1e-6:
            raise RuntimeError(f"composed safe lane mutated its cotangent: {result}")
        if unsafe["cotangent_delta_max_abs"] <= 1e-4:
            raise RuntimeError(f"composed unsafe lane lost its cotangent witness: {result}")
    for result in affine_results:
        if any(value > 2e-5 for value in result["numeric"].values()):
            raise RuntimeError(f"affine holdout did not match its eager reference: {result}")
    for result in reduction_results:
        if any(value > 2e-5 for value in result["numeric"].values()):
            raise RuntimeError(f"reduction holdout did not match its eager reference: {result}")
    conclusion = {
        "all_numeric_checks_passed": True,
        "baseline_safe_cases_passed": True,
        "composed_control_safe": True,
        "composed_candidate_violation_detected": True,
        "composed_contribution_collapse_detected": True,
        "multi_input_saved_state_cases_passed": True,
        "compiled_affine_case_passed": True,
        "reduction_cases_passed": True,
        "compiled_reduction_cases_passed": True,
        "generalization_scope": "the existing closed AOT/Triton fragment, not arbitrary Triton",
    }
    return {
        "schema": "daggertrace.local-holdout/3",
        "purpose": "blind target-free generic capture, ownership, residual-composition, reduction, and compiled-lane evidence",
        "checker": before,
        "holdout": {
            "path": HOLDOUT.relative_to(REPO).as_posix(),
            "sha256": holdout_sha256,
            "module": "local_holdout_kernel",
            "target_adapter_imported": bool(imported_adapters),
            "module_import_delta": imported_adapters,
            "shapes": list(sizes),
            "case_families": [
                "fresh-destination",
                "composed-residual-borrowed-write",
                "multi-input-saved-state",
                "reduction",
            ],
            "modules": [
                {
                    "path": HOLDOUT.relative_to(REPO).as_posix(),
                    "sha256": holdout_sha256,
                    "module": "local_holdout_kernel",
                },
                {
                    "path": AFFINE_HOLDOUT.relative_to(REPO).as_posix(),
                    "sha256": affine_holdout_sha256,
                    "module": "local_affine_holdout",
                },
                {
                    "path": REDUCTION_HOLDOUT.relative_to(REPO).as_posix(),
                    "sha256": reduction_holdout_sha256,
                    "module": "local_reduction_holdout",
                },
            ],
        },
        "cases": results,
        "composed_cases": composed_results,
        "affine_cases": affine_results,
        "reduction_cases": reduction_results,
        "conclusion": conclusion,
        "proof_valid": _proof_valid(conclusion),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[257, 4097])
    args = parser.parse_args()
    if not args.sizes or any(
        type(size) is not int or not 1 <= size <= MAX_HOLDOUT_SIZE for size in args.sizes
    ):
        parser.error(f"sizes must be integers between 1 and {MAX_HOLDOUT_SIZE}")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("sizes must not contain duplicates")
    result = collect(tuple(args.sizes))
    write_canonical_json(args.output, result)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "baseline_cases": len(result["cases"]),
                "composed_cases": len(result["composed_cases"]),
                "affine_cases": len(result["affine_cases"]),
                "reduction_cases": len(result["reduction_cases"]),
            },
            sort_keys=True,
        )
    )
    proof_valid = _proof_valid(result.get("conclusion")) and result.get("proof_valid") is True
    return 0 if proof_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
