from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from daggertrace.errors import TargetError
from daggertrace.memory_model_target import _classify_source
from daggertrace.memory_model_target import audit_triton_layer_norm
from daggertrace.replay import ReplayError
from daggertrace.triton_protocols import UNKNOWN_MEMORY_MODEL_DETAIL
from daggertrace.replay_source import _source_effects


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "external" / "triton"
PROTOCOL_OPERATIONS = (
    "atomic_add",
    "atomic_and",
    "atomic_cas",
    "atomic_max",
    "atomic_min",
    "atomic_or",
    "atomic_xchg",
    "atomic_xor",
    "barrier",
    "debug_barrier",
)


def test_pinned_tutorial_backward_reaches_memory_model_unknown():
    result = audit_triton_layer_norm(FIXTURE_ROOT)

    assert result["status"] == "UNKNOWN:MEMORY_MODEL"
    assert result["outcome"] == "UNKNOWN"
    assert result["reason"] == "MEMORY_MODEL"
    assert result["schema"] == "daggertrace.source-bound-corpus/2"
    assert "candidate" not in result
    assert "control" not in result
    assert result["target"]["repository"] == "https://github.com/triton-lang/triton"
    assert result["target"]["revision"] == "f797708c0626e5f9840ca5b0a98790e2c7cb09ad"
    assert result["target"]["files"] == {
        "source": {
            "path": "python/tutorials/05-layer-norm.py",
            "sha256": "31cda7bf7bb9b637ea662b789188d37ff5981c3973626d4e2a293db633f3bb87",
        },
        "license": {
            "path": "LICENSE",
            "sha256": "92640fb97222fd0a698ff28ce0c3782c172623f8d6c609b557636a80f28fb946",
        },
    }
    assert result["target"]["framework"] == {"triton": "3.7.1", "torch": "2.12.1"}
    assert result["provenance"] == {
        "source_bytes_verified": True,
        "license_bytes_verified": True,
        "remote_revision_authenticated": False,
    }

    evidence = result["evidence"]
    assert evidence["backward_entry"] == "LayerNorm.backward"
    assert evidence["reachable_kernels"] == [
        "_layer_norm_bwd_dx_fused",
        "_layer_norm_bwd_dwdb",
    ]
    assert evidence["backward_path"] == [
        {
            "caller": "LayerNorm.backward",
            "callee": "_layer_norm_bwd_dx_fused",
            "line": 282,
        },
        {
            "caller": "LayerNorm.backward",
            "callee": "_layer_norm_bwd_dwdb",
            "line": 290,
        },
    ]
    assert evidence["memory_model_kinds"] == [
        "atomic_cas",
        "atomic_xchg",
        "barrier",
        "unbounded_spin",
    ]
    assert [(event["kind"], event["line"]) for event in evidence["events"]] == [
        ("atomic_cas", 177),
        ("unbounded_spin", 177),
        ("atomic_xchg", 182),
        ("barrier", 191),
        ("atomic_xchg", 194),
    ]


@pytest.mark.parametrize(
    "relative",
    ["python/tutorials/05-layer-norm.py", "LICENSE"],
)
def test_pinned_tutorial_requires_exact_source_and_license_bytes(tmp_path, relative):
    root = tmp_path / "triton"
    shutil.copytree(FIXTURE_ROOT, root)
    path = root / relative
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(TargetError, match=f"hash mismatch for {relative}"):
        audit_triton_layer_norm(root)


def test_unknown_pure_or_effect_call_remains_fail_closed():
    source = """\
def _layer_norm_bwd_dx_fused(pointer):
    evil(pointer)

def _layer_norm_bwd_dwdb(pointer):
    pass

class LayerNorm:
    @staticmethod
    def backward(ctx, dy):
        _layer_norm_bwd_dx_fused[0](dy)
        _layer_norm_bwd_dwdb[0](dy)
"""

    result = _classify_source(source)

    assert result["status"] == "UNKNOWN:SOURCE_PROFILE"
    assert result["outcome"] == "UNKNOWN"
    assert result["evidence"]["unknown_calls"] == [
        {"operation": "evil", "kernel": "_layer_norm_bwd_dx_fused", "line": 2}
    ]


def test_partial_cas_loop_stays_at_the_memory_model_boundary():
    source = """\
import tl

def _layer_norm_bwd_dx_fused(pointer):
    while tl.atomic_cas(pointer, 0, 1) == 1:
        count += 1

def _layer_norm_bwd_dwdb(pointer):
    pass

class LayerNorm:
    @staticmethod
    def backward(ctx, dy):
        _layer_norm_bwd_dx_fused[0](dy)
        _layer_norm_bwd_dwdb[0](dy)
"""

    result = _classify_source(source)

    assert result["status"] == "UNKNOWN:MEMORY_MODEL"
    assert result["evidence"]["memory_model_kinds"] == ["atomic_cas"]
    assert all(event["kind"] != "unbounded_spin" for event in result["evidence"]["events"])


def test_memory_model_evidence_does_not_claim_a_bound_spin_or_a_defect():
    result = audit_triton_layer_norm(FIXTURE_ROOT)

    assert result["status"] == "UNKNOWN:MEMORY_MODEL"
    assert result["evidence"]["events"][1] == {
        "kind": "unbounded_spin",
        "operation": "while tl.atomic_cas(...): pass",
        "kernel": "_layer_norm_bwd_dx_fused",
        "line": 177,
    }
    assert result["detail"] == UNKNOWN_MEMORY_MODEL_DETAIL
    assert "bug" not in result["detail"]
    assert "mismatch" not in result["detail"]


def _adapter_protocol_source(operations: tuple[str, ...], *, loop: bool = False) -> str:
    calls = "\n".join(f"        tl.{operation}(pointer)" for operation in operations)
    body = f"    while True:\n{calls}\n        break" if loop else calls.replace("        ", "    ")
    return f'''\
def _layer_norm_bwd_dx_fused(pointer):
{body}

def _layer_norm_bwd_dwdb(pointer):
    pass

class LayerNorm:
    @staticmethod
    def backward(ctx, dy):
        _layer_norm_bwd_dx_fused[0](dy)
        _layer_norm_bwd_dwdb[0](dy)
'''


def _replay_protocol_source(operations: tuple[str, ...], *, loop: bool = False) -> str:
    calls = "\n".join(f"        tl.{operation}(pointer)" for operation in operations)
    body = f"    while True:\n{calls}\n        break" if loop else calls.replace("        ", "    ")
    return f"def protocol(pointer):\n{body}\n"


def _adapter_closed_operation_source(expression: str) -> str:
    return f'''\
def _layer_norm_bwd_dx_fused(pointer):
    value = tl.load(pointer)
    value = {expression}
    tl.store(pointer, value)

def _layer_norm_bwd_dwdb(pointer):
    pass

class LayerNorm:
    @staticmethod
    def backward(ctx, dy):
        _layer_norm_bwd_dx_fused[0](dy)
        _layer_norm_bwd_dwdb[0](dy)
'''


def _replay_closed_operation_source(expression: str) -> str:
    return f"def protocol(pointer):\n    value = tl.load(pointer)\n    value = {expression}\n    tl.store(pointer, value)\n"


def _adapter_shadowed_bare_call_source(operation: str) -> str:
    return f'''\
def _layer_norm_bwd_dx_fused(pointer):
    {operation} = pointer
    value = tl.load(pointer)
    value = {operation}(value)
    tl.store(pointer, value)

def _layer_norm_bwd_dwdb(pointer):
    pass

class LayerNorm:
    @staticmethod
    def backward(ctx, dy):
        _layer_norm_bwd_dx_fused[0](dy)
        _layer_norm_bwd_dwdb[0](dy)
'''


def _adapter_shadowed_tl_source(binding: str) -> str:
    signature = "pointer, tl" if binding == "formal" else "pointer"
    assignment = "    tl = pointer\n" if binding == "local" else ""
    return f'''\
def _layer_norm_bwd_dx_fused({signature}):
{assignment}    tl.atomic_cas(pointer, 0, 1)

def _layer_norm_bwd_dwdb(pointer):
    pass

class LayerNorm:
    @staticmethod
    def backward(ctx, dy):
        _layer_norm_bwd_dx_fused[0](dy)
        _layer_norm_bwd_dwdb[0](dy)
'''


def _replay_shadowed_tl_source(binding: str) -> str:
    signature = "pointer, tl" if binding == "formal" else "pointer"
    assignment = "    tl = pointer\n" if binding == "local" else ""
    return f"def protocol({signature}):\n{assignment}    tl.atomic_cas(pointer, 0, 1)\n"


@pytest.mark.parametrize("operation", PROTOCOL_OPERATIONS)
def test_adapter_and_replay_agree_on_every_isolated_protocol_operation(operation):
    adapter = _classify_source(_adapter_protocol_source((operation,)))

    assert adapter["status"] == "UNKNOWN:MEMORY_MODEL"
    assert adapter["detail"] == UNKNOWN_MEMORY_MODEL_DETAIL
    with pytest.raises(ReplayError) as raised:
        _source_effects(_replay_protocol_source((operation,)), {"pointer"})
    assert str(raised.value) == UNKNOWN_MEMORY_MODEL_DETAIL


@pytest.mark.parametrize(
    "operations",
    [("atomic_cas",), ("atomic_xchg", "barrier")],
    ids=["cas-only", "partial-lock-barrier"],
)
def test_adapter_and_replay_agree_on_partial_protocols(operations):
    adapter = _classify_source(_adapter_protocol_source(operations, loop=True))

    assert adapter["status"] == "UNKNOWN:MEMORY_MODEL"
    assert adapter["detail"] == UNKNOWN_MEMORY_MODEL_DETAIL
    with pytest.raises(ReplayError) as raised:
        _source_effects(_replay_protocol_source(operations, loop=True), {"pointer"})
    assert str(raised.value) == UNKNOWN_MEMORY_MODEL_DETAIL


def test_adapter_and_replay_agree_on_non_protocol_control_flow():
    adapter = _classify_source(_adapter_protocol_source(("load",), loop=True))

    assert adapter["status"] == "UNKNOWN:SOURCE_PROFILE"
    with pytest.raises(ReplayError, match="SOURCE_PROFILE"):
        _source_effects(_replay_protocol_source(("load",), loop=True), {"pointer"})


@pytest.mark.parametrize(
    "operation",
    ("exp", "exp2", "expand_dims", "join", "log2", "max", "split", "static_range"),
)
def test_adapter_and_replay_share_the_closed_tl_call_set(operation):
    expression = f"tl.{operation}(value)"
    adapter = _classify_source(_adapter_closed_operation_source(expression))

    assert adapter["status"] == "UNKNOWN:SOURCE_PROFILE"
    assert "unknown_calls" not in adapter["evidence"]
    arguments, reads, writes = _source_effects(
        _replay_closed_operation_source(expression), {"pointer"}
    )
    assert arguments == ("pointer",)
    assert reads == ("pointer",)
    assert writes == ("pointer",)


@pytest.mark.parametrize("expression", ("float(value)", "min(value, 1)", "range(0, 1)", "rsqrt(value)"))
def test_adapter_and_replay_share_the_closed_bare_call_set(expression):
    adapter = _classify_source(_adapter_closed_operation_source(expression))

    assert adapter["status"] == "UNKNOWN:SOURCE_PROFILE"
    assert "unknown_calls" not in adapter["evidence"]
    assert _source_effects(_replay_closed_operation_source(expression), {"pointer"}) == (
        ("pointer",),
        ("pointer",),
        ("pointer",),
    )


@pytest.mark.parametrize("operation", PROTOCOL_OPERATIONS)
def test_bare_protocol_names_stay_outside_the_memory_model_boundary(operation):
    expression = f"{operation}(pointer, 0, 1)"
    adapter = _classify_source(_adapter_closed_operation_source(expression))

    assert adapter["status"] == "UNKNOWN:SOURCE_PROFILE"
    assert adapter["evidence"]["unknown_calls"] == [
        {
            "operation": operation,
            "kernel": "_layer_norm_bwd_dx_fused",
            "line": 3,
        }
    ]
    with pytest.raises(ReplayError) as raised:
        _source_effects(_replay_closed_operation_source(expression), {"pointer"})
    assert str(raised.value).startswith(
        f"call {operation} is outside the supported source subset"
    )


@pytest.mark.parametrize(
    ("expression", "accepted"),
    (("value.to(tl.float32)", True), ("unknown.to(tl.float32)", False)),
)
def test_adapter_and_replay_agree_on_to_receiver_closure(expression, accepted):
    adapter = _classify_source(_adapter_closed_operation_source(expression))

    assert adapter["status"] == "UNKNOWN:SOURCE_PROFILE"
    if accepted:
        assert "unknown_calls" not in adapter["evidence"]
        assert _source_effects(
            _replay_closed_operation_source(expression), {"pointer"}
        ) == (("pointer",), ("pointer",), ("pointer",))
    else:
        assert adapter["evidence"]["unknown_calls"] == [
            {
                "operation": ".to",
                "kernel": "_layer_norm_bwd_dx_fused",
                "line": 3,
            }
        ]
        with pytest.raises(ReplayError) as raised:
            _source_effects(_replay_closed_operation_source(expression), {"pointer"})
        assert str(raised.value).startswith(
            "SOURCE_PROFILE: .to receiver is not a supported Triton expression"
        )


@pytest.mark.parametrize("operation", ("float", "min", "range", "rsqrt"))
def test_adapter_and_replay_reject_shadowed_bare_calls(operation):
    adapter = _classify_source(_adapter_shadowed_bare_call_source(operation))

    assert adapter["status"] == "UNKNOWN:SOURCE_PROFILE"
    assert adapter["evidence"]["unknown_calls"] == [
        {
            "operation": operation,
            "kernel": "_layer_norm_bwd_dx_fused",
            "line": 4,
        }
    ]
    source = _replay_closed_operation_source(f"{operation}(value)")
    source = source.replace("def protocol(pointer):", f"def protocol(pointer):\n    {operation} = pointer")
    with pytest.raises(ReplayError) as raised:
        _source_effects(source, {"pointer"})
    assert str(raised.value) == (
        f"SOURCE_PROFILE: supported callable {operation} is shadowed locally"
    )


@pytest.mark.parametrize("binding", ("formal", "local"))
def test_adapter_and_replay_reject_shadowed_tl_protocols(binding):
    adapter = _classify_source(_adapter_shadowed_tl_source(binding))

    assert adapter["status"] == "UNKNOWN:SOURCE_PROFILE"
    assert adapter["evidence"]["unknown_calls"][0]["operation"] == "tl.atomic_cas"
    with pytest.raises(ReplayError) as raised:
        _source_effects(_replay_shadowed_tl_source(binding), {"pointer"})
    assert str(raised.value) == "SOURCE_PROFILE: the Triton namespace is shadowed locally"


def test_unshadowed_direct_tl_protocol_remains_memory_model_unknown():
    expression = "tl.atomic_cas(pointer, 0, 1)"
    adapter = _classify_source(_adapter_closed_operation_source(expression))

    assert adapter["status"] == "UNKNOWN:MEMORY_MODEL"
    assert adapter["detail"] == UNKNOWN_MEMORY_MODEL_DETAIL
    with pytest.raises(ReplayError) as raised:
        _source_effects(_replay_closed_operation_source(expression), {"pointer"})
    assert str(raised.value) == UNKNOWN_MEMORY_MODEL_DETAIL


def test_namespace_qualified_protocol_call_does_not_cross_the_tl_boundary():
    expression = "lock.atomic_cas(pointer, 0, 1)"
    adapter = _classify_source(_adapter_closed_operation_source(expression))

    assert adapter["status"] == "UNKNOWN:SOURCE_PROFILE"
    assert adapter["evidence"]["unknown_calls"] == [
        {
            "operation": "lock.atomic_cas",
            "kernel": "_layer_norm_bwd_dx_fused",
            "line": 3,
        }
    ]
    with pytest.raises(ReplayError) as raised:
        _source_effects(_replay_closed_operation_source(expression), {"pointer"})
    assert "atomic_cas" in str(raised.value)
    assert "outside the supported source subset" in str(raised.value)


def test_tl_sqrt_remains_outside_the_shared_closed_call_set():
    adapter = _classify_source(_adapter_closed_operation_source("tl.sqrt(value)"))

    assert adapter["evidence"]["unknown_calls"] == [
        {
            "operation": "tl.sqrt",
            "kernel": "_layer_norm_bwd_dx_fused",
            "line": 3,
        }
    ]
    with pytest.raises(ReplayError, match="tl.sqrt"):
        _source_effects(_replay_closed_operation_source("tl.sqrt(value)"), {"pointer"})
