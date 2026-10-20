import json
import sys
from pathlib import Path

import pytest

from evidence import collect_review
from daggertrace.memory_model_target import audit_triton_layer_norm
from daggertrace.validation_v2 import (
    ABLATION_COUNT,
    CLOSURE_MUTATION_COUNT,
    CORRUPTION_MUTATION_COUNT,
    EVIDENCE_INTEGRITY_MUTATION_COUNT,
    V2_MUTATION_COUNT,
    V2_ORACLE_CASES,
)


TRITON_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "external" / "triton"


def _inputs(root: Path) -> tuple[Path, Path, Path]:
    liger = root / "inputs" / "liger-rmsnorm" / "certificate.json"
    art = root / "inputs" / "openpipe-art-local-stats" / "certificate.json"
    triton = root / "targets" / "triton"
    for packet in (liger, art):
        packet.parent.mkdir(parents=True, exist_ok=True)
        packet.write_text(json.dumps({"packet": packet.parent.name}), encoding="utf-8")
    triton.mkdir(parents=True)
    return liger, art, triton


def _valid_source_result() -> dict[str, object]:
    return audit_triton_layer_norm(TRITON_FIXTURE_ROOT)


def _stub_collection_dependencies(monkeypatch, source, mutation_names=()):
    monkeypatch.setattr(
        collect_review,
        "run_semantic_validation",
        lambda: {
            "oracle_passed": V2_ORACLE_CASES,
            "oracle_cases": V2_ORACLE_CASES,
            "mutation_cases": V2_MUTATION_COUNT,
            "ablations": ABLATION_COUNT,
            "ablations_observed": ABLATION_COUNT,
        },
    )
    monkeypatch.setattr(collect_review, "replay_certificate", lambda _path: {"valid": True})
    monkeypatch.setattr(collect_review, "_subprocess_replay", lambda _path: {"valid": True})
    monkeypatch.setattr(
        collect_review,
        "_certificate_mutations",
        lambda _certificate: [(name, None) for name in mutation_names],
    )
    monkeypatch.setattr(
        collect_review,
        "validate_certificate_mutations",
        lambda *_args, **_kwargs: len(mutation_names),
    )
    monkeypatch.setattr(collect_review, "audit_triton_layer_norm", lambda _root: source)


def test_review_collection_does_not_encode_invocation_paths(tmp_path, monkeypatch):
    _stub_collection_dependencies(monkeypatch, _valid_source_result())

    first = collect_review.collect(*_inputs(tmp_path / "first"))
    second = collect_review.collect(*_inputs(tmp_path / "unrelated" / "second"))

    assert first == second
    assert first["semantic_gate"] == {
        "oracle_passed": 48,
        "oracle_cases": 48,
        "mutation_cases": 34,
        "ablations": 7,
        "ablations_observed": 7,
    }
    assert first["packets"]["liger"]["path"] == "liger/liger-rmsnorm/certificate.json"
    assert first["packets"]["art"]["path"] == "art/openpipe-art-local-stats/certificate.json"
    assert first["memory_model_boundary"]["target_root"] == "triton-layernorm"
    assert first["conclusion"]["generic_oracle_passed"] is True
    assert first["conclusion"]["all_mutations_killed"] is False
    assert first["proof_valid"] is False


def test_review_resolves_packet_paths_before_in_process_and_subprocess_replay(
    tmp_path, monkeypatch
):
    liger, art, triton = _inputs(tmp_path / "inputs")
    monkeypatch.chdir(tmp_path)
    relative_paths = tuple(path.relative_to(tmp_path) for path in (liger, art, triton))
    replay_paths = []

    _stub_collection_dependencies(monkeypatch, _valid_source_result())
    monkeypatch.setattr(
        collect_review,
        "replay_certificate",
        lambda path: replay_paths.append(path) or {"valid": True},
    )
    monkeypatch.setattr(
        collect_review,
        "_subprocess_replay",
        lambda path: replay_paths.append(path) or {"valid": True},
    )

    collect_review.collect(*relative_paths)

    assert replay_paths
    assert all(path.is_absolute() for path in replay_paths)


def test_review_source_gate_requires_a_fully_bound_memory_model_result(tmp_path, monkeypatch):
    mutation_names = (
        *(f"corrupt-{index}" for index in range(CORRUPTION_MUTATION_COUNT)),
        *(f"closure-{index}" for index in range(CLOSURE_MUTATION_COUNT)),
        *(f"integrity-{index}" for index in range(EVIDENCE_INTEGRITY_MUTATION_COUNT)),
    )
    assert len(mutation_names) == V2_MUTATION_COUNT

    _stub_collection_dependencies(monkeypatch, _valid_source_result(), mutation_names)
    valid = collect_review.collect(*_inputs(tmp_path / "valid"))

    assert valid["conclusion"]["memory_protocol_fails_closed"] is True
    assert valid["proof_valid"] is True

    _stub_collection_dependencies(
        monkeypatch,
        {
            "outcome": "UNKNOWN",
            "reason": "MEMORY_MODEL",
            "evidence": {
                "reachable_kernels": ["_layer_norm_bwd_dx_fused", "_layer_norm_bwd_dwdb"]
            },
        },
        mutation_names,
    )
    malformed = collect_review.collect(*_inputs(tmp_path / "malformed"))

    assert malformed["conclusion"]["memory_protocol_fails_closed"] is False
    assert malformed["proof_valid"] is False


@pytest.mark.parametrize("proof_valid,expected", ((False, 1), (None, 1), (True, 0)))
def test_review_main_propagates_proof_status(monkeypatch, tmp_path, proof_valid, expected):
    monkeypatch.setattr(
        collect_review,
        "collect",
        lambda *_args: {
            "mutation_attack": {"killed": 0},
            "conclusion": {"proof_gate": proof_valid},
            "proof_valid": proof_valid,
        },
    )
    monkeypatch.setattr(collect_review, "write_canonical_json", lambda *_args: None)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_review",
            "--liger",
            str(tmp_path / "liger.json"),
            "--art",
            str(tmp_path / "art.json"),
            "--triton-root",
            str(tmp_path / "triton"),
            "--output",
            str(tmp_path / "review.json"),
        ],
    )
    assert collect_review.main() == expected


def test_checked_review_contract_remains_a_v2_evidence_record():
    root = Path(__file__).resolve().parents[1]
    review = json.loads(
        (root / "evidence/results/bundle/review.json").read_text(encoding="utf-8")
    )
    assert review["schema"] == "daggertrace.local-review/1"
    assert review["semantic_gate"]["oracle_passed"] == 48
    assert review["semantic_gate"]["mutation_cases"] == 34
    assert review["semantic_gate"]["ablations_observed"] == 7
