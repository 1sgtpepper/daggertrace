"""Collect independent replay, mutation, and source-boundary evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from daggertrace.canonical import write_canonical_json  # noqa: E402
from daggertrace.cli_replay import _replay_mutation  # noqa: E402
from daggertrace.errors import TargetError  # noqa: E402
from daggertrace.memory_model_target import (  # noqa: E402
    _validate_source_result,
    audit_triton_layer_norm,
)
from daggertrace.replay import replay_certificate  # noqa: E402
from daggertrace.validation import run_semantic_validation  # noqa: E402
from daggertrace.validation_mutations import (  # noqa: E402
    _certificate_mutations,
    validate_certificate_mutations,
)
from daggertrace.validation_v2 import (  # noqa: E402
    ABLATION_COUNT,
    CLOSURE_MUTATION_COUNT,
    CORRUPTION_MUTATION_COUNT,
    EVIDENCE_INTEGRITY_MUTATION_COUNT,
    V2_MUTATION_COUNT,
    V2_ORACLE_CASES,
)


def _subprocess_replay(path: Path) -> dict[str, Any]:
    path = path.resolve()
    command = [sys.executable, "-B", "-m", "daggertrace", "replay", str(path), "--json"]
    completed = subprocess.run(
        command,
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def collect(liger: Path, art: Path, triton_root: Path) -> dict[str, Any]:
    liger = liger.resolve()
    art = art.resolve()
    triton_root = triton_root.resolve()
    semantic = run_semantic_validation()
    liger_replay = replay_certificate(liger)
    art_replay = replay_certificate(art)
    with liger.open("r", encoding="utf-8") as handle:
        certificate = json.load(handle)
    mutation_names = [name for name, _ in _certificate_mutations(certificate)]
    killed = validate_certificate_mutations(
        certificate,
        liger,
        replay_fn=_replay_mutation,
    )
    source = audit_triton_layer_norm(triton_root)
    try:
        _validate_source_result(source)
    except TargetError:
        source_gate = False
    else:
        source_gate = True
    subprocess_replay = {
        "liger": _subprocess_replay(liger),
        "art": _subprocess_replay(art),
    }
    with tempfile.TemporaryDirectory(prefix="daggertrace-evidence-replay-") as temporary:
        relocated = Path(temporary)
        shutil.copytree(liger.parent, relocated / "liger")
        shutil.copytree(art.parent, relocated / "art")
        relocated_replay = {
            "liger": _subprocess_replay(relocated / "liger" / liger.name),
            "art": _subprocess_replay(relocated / "art" / art.name),
        }
    semantic_gate = (
        isinstance(semantic, dict)
        and type(semantic.get("oracle_cases")) is int
        and type(semantic.get("oracle_passed")) is int
        and semantic["oracle_cases"] == V2_ORACLE_CASES
        and semantic["oracle_passed"] == V2_ORACLE_CASES
        and type(semantic.get("mutation_cases")) is int
        and type(semantic.get("ablations")) is int
        and type(semantic.get("ablations_observed")) is int
        and semantic["mutation_cases"] == V2_MUTATION_COUNT
        and semantic["ablations"] == ABLATION_COUNT
        and semantic["ablations_observed"] == ABLATION_COUNT
    )
    mutation_categories = {
        "corruption": CORRUPTION_MUTATION_COUNT,
        "closure": CLOSURE_MUTATION_COUNT,
        "evidence_integrity": EVIDENCE_INTEGRITY_MUTATION_COUNT,
    }
    mutation_gate = (
        type(killed) is int
        and killed >= 0
        and type(mutation_names) is list
        and all(type(name) is str and name for name in mutation_names)
        and len(mutation_names) == V2_MUTATION_COUNT
        and len(set(mutation_names)) == len(mutation_names)
        and killed == len(mutation_names)
        and all(type(value) is int and value >= 0 for value in mutation_categories.values())
        and sum(mutation_categories.values()) == V2_MUTATION_COUNT
        and sum(name.startswith("corrupt-") for name in mutation_names)
        == mutation_categories["corruption"]
        and sum(name.startswith("closure-") for name in mutation_names)
        == mutation_categories["closure"]
        and sum(name.startswith("integrity-") for name in mutation_names)
        == mutation_categories["evidence_integrity"]
    )

    def replay_gate(values: dict[str, Any]) -> bool:
        return (
            set(values) == {"liger", "art"}
            and all(isinstance(result, dict) and result.get("valid") is True for result in values.values())
        )

    conclusion = {
        "generic_oracle_passed": semantic_gate,
        "all_mutations_killed": mutation_gate,
        "both_packets_replayed_in_process": replay_gate(
            {"liger": liger_replay, "art": art_replay}
        ),
        "both_packets_replayed_in_clean_subprocess": replay_gate(subprocess_replay),
        "both_packets_replayed_after_relocation": replay_gate(relocated_replay),
        "memory_protocol_fails_closed": source_gate,
    }
    return {
        "schema": "daggertrace.local-review/1",
        "purpose": "independent generic replay, semantic oracle, mutation, and memory-model evidence",
        "semantic_gate": semantic,
        "packets": {
            "liger": {
                "path": "liger/liger-rmsnorm/certificate.json",
                "replay_in_process": liger_replay,
                "replay_subprocess": subprocess_replay["liger"],
                "replay_relocated_subprocess": relocated_replay["liger"],
            },
            "art": {
                "path": "art/openpipe-art-local-stats/certificate.json",
                "replay_in_process": art_replay,
                "replay_subprocess": subprocess_replay["art"],
                "replay_relocated_subprocess": relocated_replay["art"],
            },
        },
        "mutation_attack": {
            "packet": "liger/liger-rmsnorm/certificate.json",
            "killed": killed,
            "total": V2_MUTATION_COUNT,
            "categories": mutation_categories,
            "names": mutation_names,
            "replay_path": "generic replay plus target-owned closure validator",
        },
        "memory_model_boundary": {
            "target_root": "triton-layernorm",
            "result": source,
            "materialized_aot_packet": False,
            "reason": "the pinned source reaches two backward kernels while the source-only /2 wire form admits one",
        },
        "conclusion": {
            **conclusion,
        },
        "proof_valid": bool(conclusion) and all(value is True for value in conclusion.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liger", type=Path, required=True)
    parser.add_argument("--art", type=Path, required=True)
    parser.add_argument("--triton-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.liger, args.art, args.triton_root)
    write_canonical_json(args.output, result)
    print(json.dumps({"output": args.output.as_posix(), "mutations": result["mutation_attack"]["killed"]}, sort_keys=True))
    conclusion = result.get("conclusion")
    conclusion_valid = (
        isinstance(conclusion, dict)
        and bool(conclusion)
        and all(value is True for value in conclusion.values())
    )
    proof_valid = conclusion_valid and result.get("proof_valid") is True
    return 0 if proof_valid is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
