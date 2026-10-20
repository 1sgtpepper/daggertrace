from __future__ import annotations

import copy
import json
from pathlib import Path

from daggertrace.canonical import sha256_bytes
from daggertrace.certificate import make_certificate, write_certificate
from daggertrace.model import Capture
from daggertrace.ownership import analyze
from daggertrace.validation_v2_fixtures import _micrograph

GENERIC_REPRODUCER = b"# packet-local reproducer\n"


def _write_canonical(path: Path, value) -> None:
    path.write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        + b"\n"
    )


def _packet(
    tmp_path: Path, *, paired: bool = False, alias_destination: bool = False
) -> tuple[Path, dict]:
    candidate, control = _micrograph(alias_destination, "direct", False, 1, False)
    candidate = _with_diagnostics(candidate)
    control = _with_diagnostics(control)
    control_capture = control if paired else None
    source = candidate.kernels[0].source.encode()
    source_path = tmp_path / "external" / "source" / "kernel.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source)
    zero = {"output_max_abs": 0.0, "gx_max_abs": 0.0, "gw_max_abs": 0.0}
    target = {
        "repository": "https://example.invalid/repository",
        "revision": "a" * 40,
        "files": [
            {
                "packet_path": "external/source/kernel.py",
                "target_path": "src/kernel.py",
                "size": len(source),
                "sha256": sha256_bytes(source),
                "role": "source",
            }
        ],
        "kernels": [
            {
                "module": candidate.kernels[0].module,
                "qualname": candidate.kernels[0].qualname,
                "file": "external/source/kernel.py",
                "source_sha256": candidate.kernels[0].source_sha256,
            }
        ],
        "specialization": {
            "shape": [2, 2],
            "case": "safe",
            "numerical_observations": {
                "candidate_vs_reference": zero,
                "candidate_vs_control": None,
                "seed": 0,
            },
        },
    }
    lane = {
        "device_ms": [1.0, 2.0, 3.0],
        "wall_ms": [2.0, 3.0, 4.0],
        "device_median_ms": 2.0,
        "wall_median_ms": 3.0,
        "device_mad_ms": 1.0,
    }
    runtime = {
        "environment": {
            "python": "3.14.6",
            "platform": "test",
            "torch": "2.12.1+cu130",
            "torch_git": "7269437d655783a26cba32aa88195b741ff496aa",
            "triton": "3.7.1",
            "cuda_runtime": "13.0",
            "device": "test-device",
            "capability": [1, 0],
            "total_memory_bytes": 0,
            "multiprocessors": 0,
            "driver": None,
        },
        "candidate": {"compiled": None},
        "control": None,
    }
    certificate = make_certificate(
        candidate=candidate,
        control=control_capture,
        analysis=analyze(candidate, control_capture),
        target=target,
        runtime=runtime,
        benchmark={
            "warmups": 1,
            "samples": 3,
            "lanes": {"candidate": lane},
            "claim": "supporting evidence only",
        },
        controls={},
        validation={
            "oracle_cases": 0,
            "oracle_passed": 0,
            "mutation_cases": 0,
            "mutations_killed": 0,
            "ablations": 0,
            "ablations_observed": 0,
        },
    )
    certificate_path = tmp_path / "certificate.json"
    write_certificate(certificate_path, certificate, reproducer_bytes=GENERIC_REPRODUCER)
    return certificate_path, certificate


def _with_diagnostics(capture: Capture) -> Capture:
    raw = capture.to_dict()
    raw["diagnostics"] = {
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
    return Capture.from_dict(raw)


def _with_second_differentiable_input(capture: Capture) -> Capture:
    raw = capture.to_dict()
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
    return Capture.from_dict(raw)
