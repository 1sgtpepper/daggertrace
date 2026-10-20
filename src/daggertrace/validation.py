"""Semantic tests that run without PyTorch or Triton."""

from __future__ import annotations

from .ownership import analyze
from .validation_mutations import validate_certificate_mutations as validate_certificate_mutations
from .validation_v2 import (
    ABLATION_COUNT,
    V2_MUTATION_COUNT,
    V2_ORACLE_CASES,
    _matches_expected,
    _run_ablations,
    _run_kernel_identity_invariance,
    _semantic_cases,
)


def run_semantic_validation() -> dict[str, int]:
    oracle_passed = 0
    for name, candidate, control, expected in _semantic_cases():
        result = analyze(candidate, control)
        if not _matches_expected(result, expected, candidate, control):
            raise AssertionError(
                "semantic oracle mismatch: "
                f"case={name}, expected={expected}, result={result.to_dict()}"
            )
        oracle_passed += 1
    oracle_passed += _run_kernel_identity_invariance(analyze_fn=analyze)
    ablations_observed = _run_ablations(analyze_fn=analyze)
    if oracle_passed != V2_ORACLE_CASES or ablations_observed != ABLATION_COUNT:
        raise AssertionError("validation denominator changed")
    return {
        "oracle_cases": V2_ORACLE_CASES,
        "oracle_passed": oracle_passed,
        "mutation_cases": V2_MUTATION_COUNT,
        "ablations": ABLATION_COUNT,
        "ablations_observed": ablations_observed,
    }
