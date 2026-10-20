"""Source-bound evidence for the pinned Triton LayerNorm memory-model boundary.

This adapter is deliberately narrower than the ownership core.  It records why the
official tutorial cannot enter the closed AOT/Triton ownership language; it does not
approximate atomics, rewrite the spin loop, or claim a numerical or compiler defect.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .effects import _assignments, _closed_local_bindings, _closed_value, _local_bindings
from .errors import TargetError
from .source_files import _read_verified_tree_file
from .triton_protocols import (
    MEMORY_MODEL_REASON,
    MEMORY_MODEL_STATUS,
    TRITON_PROTOCOL_OPERATIONS,
    _TRITON_SOURCE_BARE_CALLS,
    _TRITON_SOURCE_TL_OPERATIONS,
    UNKNOWN_MEMORY_MODEL_DETAIL,
    triton_protocol_kind,
)


TRITON_REPOSITORY = "https://github.com/triton-lang/triton"
TRITON_REVISION = "f797708c0626e5f9840ca5b0a98790e2c7cb09ad"
TRITON_SOURCE_PATH = "python/tutorials/05-layer-norm.py"
TRITON_SOURCE_SHA256 = "31cda7bf7bb9b637ea662b789188d37ff5981c3973626d4e2a293db633f3bb87"
TRITON_LICENSE_PATH = "LICENSE"
TRITON_LICENSE_SHA256 = "92640fb97222fd0a698ff28ce0c3782c172623f8d6c609b557636a80f28fb946"
_SOURCE_CORPUS_SCHEMA = "daggertrace.source-bound-corpus/2"

_BACKWARD_KERNELS = ("_layer_norm_bwd_dx_fused", "_layer_norm_bwd_dwdb")
_KNOWN_KERNEL_CALLS = {
    *(f"tl.{operation}" for operation in _TRITON_SOURCE_TL_OPERATIONS),
    *_TRITON_SOURCE_BARE_CALLS,
    ".to",
    *(f"tl.{operation}" for operation in TRITON_PROTOCOL_OPERATIONS),
}


def audit_triton_layer_norm(source_root: Path) -> dict[str, Any]:
    """Audit the pinned tutorial bytes and return a closed UNKNOWN result.

    The returned result is source-bound corpus evidence only, not an ownership
    certificate or a capture.  ``remote_revision_authenticated`` is explicitly false:
    Repository and revision fields are provenance metadata, not a cryptographic
    statement about remote history.  No AOT/Inductor occurrence may be implied by a
    source-level memory-model rejection.
    """

    source = _read_verified_tree_file(
        source_root, TRITON_SOURCE_PATH, TRITON_SOURCE_SHA256, "Triton"
    )
    _read_verified_tree_file(source_root, TRITON_LICENSE_PATH, TRITON_LICENSE_SHA256, "Triton")
    try:
        source_text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TargetError("pinned Triton tutorial is not UTF-8") from exc

    result = _classify_source(source_text)
    result["schema"] = _SOURCE_CORPUS_SCHEMA
    result["target"] = {
        "repository": TRITON_REPOSITORY,
        "revision": TRITON_REVISION,
        "files": {
            "source": {"path": TRITON_SOURCE_PATH, "sha256": TRITON_SOURCE_SHA256},
            "license": {"path": TRITON_LICENSE_PATH, "sha256": TRITON_LICENSE_SHA256},
        },
        "framework": {"triton": "3.7.1", "torch": "2.12.1"},
    }
    result["provenance"] = {
        "source_bytes_verified": True,
        "license_bytes_verified": True,
        "remote_revision_authenticated": False,
    }
    return result


def _validate_source_result(result: Any) -> dict[str, Any]:
    """Validate the fixed source classification exposed by the inspection route."""

    expected_keys = {
        "outcome",
        "reason",
        "status",
        "detail",
        "schema",
        "target",
        "provenance",
        "evidence",
    }
    if not isinstance(result, dict):
        raise TargetError("triton-layernorm source classification result is not an object")
    if set(result) != expected_keys or any(
        result.get(key) != expected
        for key, expected in (
            ("schema", _SOURCE_CORPUS_SCHEMA),
            ("outcome", "UNKNOWN"),
            ("reason", MEMORY_MODEL_REASON),
            ("status", MEMORY_MODEL_STATUS),
            ("detail", UNKNOWN_MEMORY_MODEL_DETAIL),
        )
    ):
        raise TargetError(
            "triton-layernorm source classification is not the fixed memory-model result"
        )
    expected_target = {
        "repository": TRITON_REPOSITORY,
        "revision": TRITON_REVISION,
        "files": {
            "source": {"path": TRITON_SOURCE_PATH, "sha256": TRITON_SOURCE_SHA256},
            "license": {"path": TRITON_LICENSE_PATH, "sha256": TRITON_LICENSE_SHA256},
        },
        "framework": {"triton": "3.7.1", "torch": "2.12.1"},
    }
    if result["target"] != expected_target:
        raise TargetError("triton-layernorm source classification target manifest is malformed")
    expected_evidence = {
        "backward_entry": "LayerNorm.backward",
        "backward_path": [
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
        ],
        "reachable_kernels": list(_BACKWARD_KERNELS),
        "memory_model_kinds": [
            "atomic_cas",
            "atomic_xchg",
            "barrier",
            "unbounded_spin",
        ],
        "events": [
            {
                "kind": "atomic_cas",
                "operation": "tl.atomic_cas",
                "kernel": "_layer_norm_bwd_dx_fused",
                "line": 177,
            },
            {
                "kind": "unbounded_spin",
                "operation": "while tl.atomic_cas(...): pass",
                "kernel": "_layer_norm_bwd_dx_fused",
                "line": 177,
            },
            {
                "kind": "atomic_xchg",
                "operation": "tl.atomic_xchg",
                "kernel": "_layer_norm_bwd_dx_fused",
                "line": 182,
            },
            {
                "kind": "barrier",
                "operation": "tl.debug_barrier",
                "kernel": "_layer_norm_bwd_dx_fused",
                "line": 191,
            },
            {
                "kind": "atomic_xchg",
                "operation": "tl.atomic_xchg",
                "kernel": "_layer_norm_bwd_dx_fused",
                "line": 194,
            },
        ],
    }
    if result["evidence"] != expected_evidence:
        raise TargetError("triton-layernorm source classification evidence is malformed")
    if result["provenance"] != {
        "source_bytes_verified": True,
        "license_bytes_verified": True,
        "remote_revision_authenticated": False,
    }:
        raise TargetError("triton-layernorm source classification provenance is malformed")
    return result


def _classify_source(source: str) -> dict[str, Any]:
    """Classify source without turning unsupported calls into acceptance."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return _unknown("SOURCE_PROFILE", f"tutorial source is not parseable Python: {exc}")

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    backward = _find_layer_norm_backward(tree)
    if backward is None:
        return _unknown("SOURCE_PROFILE", "tutorial lacks LayerNorm.backward")

    calls = [
        (name, call.lineno)
        for call in ast.walk(backward)
        if isinstance(call, ast.Call)
        for name in [_called_name(call.func)]
        if name is not None
    ]
    reachable = tuple(name for name in _BACKWARD_KERNELS if name in {name for name, _ in calls})
    path = [
        {"caller": "LayerNorm.backward", "callee": name, "line": line}
        for name, line in calls
        if name in _BACKWARD_KERNELS
    ]
    if reachable != _BACKWARD_KERNELS:
        return _unknown(
            "SOURCE_PROFILE",
            "LayerNorm.backward does not reach the complete pinned backward kernel path",
            {"backward_path": path, "reachable_kernels": list(reachable)},
        )

    reachable_functions = [functions[name] for name in reachable if name in functions]
    if len(reachable_functions) != len(reachable):
        return _unknown(
            "SOURCE_PROFILE",
            "LayerNorm.backward reaches a kernel definition outside the supplied source",
            {"backward_path": path, "reachable_kernels": list(reachable)},
        )

    events: list[dict[str, Any]] = []
    unknown_calls: list[dict[str, Any]] = []
    protocol_present = False
    for function in reachable_functions:
        arguments = {argument.arg for argument in function.args.args}
        local_bindings = _local_bindings(function)
        assignments = _assignments(function)
        closed = arguments | _closed_local_bindings(function, arguments)
        namespace_shadowed = "tl" in local_bindings
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                operation = _operation_name(node.func)
                if isinstance(node.func, ast.Name) and operation in _TRITON_SOURCE_BARE_CALLS:
                    if operation not in local_bindings:
                        continue
                    unknown_calls.append(
                        {"operation": operation, "kernel": function.name, "line": node.lineno}
                    )
                    continue
                if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
                    if not namespace_shadowed and _closed_value(
                        node.func.value, closed, assignments, set()
                    ):
                        continue
                    unknown_calls.append(
                        {"operation": operation, "kernel": function.name, "line": node.lineno}
                    )
                    continue
                direct_tl = (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "tl"
                )
                kind = (
                    triton_protocol_kind(f"tl.{node.func.attr}")
                    if direct_tl and not namespace_shadowed
                    else None
                )
                if kind is not None:
                    protocol_present = True
                    events.append(
                        {
                            "kind": kind,
                            "operation": operation,
                            "kernel": function.name,
                            "line": node.lineno,
                        }
                    )
                    continue
                if operation not in _KNOWN_KERNEL_CALLS:
                    unknown_calls.append(
                        {"operation": operation, "kernel": function.name, "line": node.lineno}
                    )
                elif direct_tl and namespace_shadowed:
                    unknown_calls.append(
                        {"operation": operation, "kernel": function.name, "line": node.lineno}
                    )
            elif (
                isinstance(node, ast.While)
                and not namespace_shadowed
                and _is_unbounded_cas_spin(node)
            ):
                events.append(
                    {
                        "kind": "unbounded_spin",
                        "operation": "while tl.atomic_cas(...): pass",
                        "kernel": function.name,
                        "line": node.lineno,
                    }
                )
    events.sort(key=lambda event: (event["line"], event["kind"], event["operation"]))
    unknown_calls.sort(key=lambda call: (call["line"], call["operation"]))
    memory_model_kinds = frozenset(event["kind"] for event in events)
    evidence = {
        "backward_entry": "LayerNorm.backward",
        "backward_path": path,
        "reachable_kernels": list(reachable),
        "memory_model_kinds": sorted(memory_model_kinds),
        "events": events,
    }
    if protocol_present:
        return _unknown(
            MEMORY_MODEL_REASON,
            UNKNOWN_MEMORY_MODEL_DETAIL,
            evidence,
        )
    if events:
        return _unknown(
            "SOURCE_PROFILE",
            "backward contains synchronization outside the pinned memory-model protocol",
            evidence,
        )
    if unknown_calls:
        return _unknown(
            "SOURCE_PROFILE",
            "backward contains calls outside the supported source subset",
            {**evidence, "unknown_calls": unknown_calls},
        )
    return _unknown(
        "SOURCE_PROFILE",
        "the source bridge has no closed memory-model-free theorem for this target",
        evidence,
    )


def _find_layer_norm_backward(tree: ast.AST) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "LayerNorm":
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == "backward":
                return child
    return None


def _is_unbounded_cas_spin(node: ast.While) -> bool:
    test = node.test
    return (
        bool(node.body)
        and all(isinstance(statement, ast.Pass) for statement in node.body)
        and isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == 1
        and isinstance(test.left, ast.Call)
        and triton_protocol_kind(_operation_name(test.left.func)) == "atomic_cas"
    )


def _called_name(function: ast.AST) -> str | None:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Subscript):
        return _called_name(function.value)
    return None


def _operation_name(function: ast.AST) -> str:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        if function.attr == "to":
            return ".to"
        receiver = _qualified_name(function.value)
        if receiver is not None:
            return f"{receiver}.{function.attr}"
        return f"<{type(function.value).__name__}>.{function.attr}"
    return type(function).__name__


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        receiver = _qualified_name(node.value)
        return f"{receiver}.{node.attr}" if receiver is not None else None
    return None


def _unknown(reason: str, detail: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "outcome": "UNKNOWN",
        "reason": reason,
        "status": f"UNKNOWN:{reason}",
        "detail": detail,
    }
    if evidence is not None:
        result["evidence"] = evidence
    return result
