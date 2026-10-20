from __future__ import annotations

import os
from pathlib import Path

import pytest

from daggertrace.liger_target import audit_liger, run_liger_reproducer


@pytest.mark.gpu
def test_exact_liger_residual_witness():
    root = os.environ.get("DAGGERTRACE_LIGER_ROOT")
    if not root:
        pytest.skip("set DAGGERTRACE_LIGER_ROOT to the exact pinned Liger checkout")
    result = run_liger_reproducer(Path(root))
    assert result["candidate_vs_reference"]["gx_max_abs"] > 1e-4
    assert result["control_vs_reference"]["gx_max_abs"] < 2e-5
    assert result["candidate_vs_control"]["output_max_abs"] == 0.0
    assert result["candidate_vs_control"]["gw_max_abs"] == 0.0


@pytest.mark.gpu
def test_exact_liger_full_audit(tmp_path):
    root = os.environ.get("DAGGERTRACE_LIGER_ROOT")
    if not root:
        pytest.skip("set DAGGERTRACE_LIGER_ROOT to the exact pinned Liger checkout")
    result = audit_liger(
        tmp_path / "audit",
        target_root=Path(root),
        warmups=1,
        samples=3,
    )
    runtime = result["runtime"]
    for lane in ("candidate", "control"):
        evidence = runtime[lane]["compiled"]["occurrence"]
        assert evidence["counter_delta"]["aot_autograd"]["total"] == 1
        assert evidence["counter_delta"]["aot_autograd"]["ok"] == 1
        assert evidence["counter_delta"]["aot_autograd"].get("not_ok", 0) == 0
        assert evidence["counter_delta"]["inductor"]["extern_calls"] == 2
        assert evidence["generated_kernel_count"] == 4
        assert evidence["ir_nodes_pre_fusion"] == 9
