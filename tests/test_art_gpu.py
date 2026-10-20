from __future__ import annotations

import os
from pathlib import Path

import pytest

from daggertrace import art_target
from daggertrace.cli import replay_certificate
from daggertrace.model import Outcome


@pytest.mark.gpu
def test_art_gpu_audit_is_candidate_only_and_records_aot_inductor_evidence(tmp_path):
    if not Path("/dev/nvidiactl").exists() and not Path("/dev/nvidia0").exists():
        pytest.skip("no NVIDIA device is available in this environment")
    root = os.environ.get("DAGGERTRACE_ART_ROOT")
    if not root:
        pytest.skip("set DAGGERTRACE_ART_ROOT to the exact pinned ART checkout")

    result = art_target.audit_art(tmp_path / "audit", target_root=Path(root), warmups=1, samples=3)

    assert result["control"] is None
    assert result["controls"] == {}
    assert result["artifacts"]["control_capture"] is None
    assert result["analysis"].outcome is Outcome.SAFE_IN_FRAGMENT
    occurrence = result["runtime"]["candidate"]["compiled"]["occurrence"]
    assert occurrence["counter_delta"]["aot_autograd"]["total"] == 1
    assert occurrence["counter_delta"]["aot_autograd"]["ok"] == 1
    assert occurrence["counter_delta"]["aot_autograd"].get("not_ok", 0) == 0
    assert occurrence["counter_delta"]["inductor"]["extern_calls"] >= 1
    assert occurrence["generated_kernel_count"] >= 1
    assert occurrence["ir_nodes_pre_fusion"] >= 1
    assert result["benchmark"]["samples"] == 3
    assert (tmp_path / "audit" / "control.capture.json").read_text() == "null\n"
    replay = replay_certificate(tmp_path / "audit" / "certificate.json")
    assert replay["generic_replay_compatible"] is True
    assert replay["adapter"] == art_target.ART_TARGET_ID
