from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from daggertrace.canonical import sha256_bytes
from daggertrace.replay import (
    MAX_EXTERNAL_FILE_BYTES,
    ReplayError,
    replay_certificate,
    replay_source_evidence,
)
from daggertrace.replay_graph import _bind_capture_kernels
from daggertrace.replay_packet import _sha, _target_manifest
from daggertrace.replay_source import _function_slice, _source_effects
from daggertrace.replay_v2 import _capture
from daggertrace.triton_protocols import UNKNOWN_MEMORY_MODEL_DETAIL
from daggertrace.validation_v2_fixtures import _micrograph

from _certificate_fixtures import _packet, _write_canonical
def test_coherent_target_relabel_preserves_the_replayed_result(tmp_path):
    certificate_path, certificate = _packet(tmp_path)
    relabeled = copy.deepcopy(certificate)
    relabeled["target"]["repository"] = "https://example.invalid/renamed"
    relabeled["target"]["revision"] = "d" * 40
    relabeled["target"]["files"][0]["target_path"] = "renamed/source.py"
    relabeled["digest"] = _sha({key: value for key, value in relabeled.items() if key != "digest"})
    _write_canonical(certificate_path, relabeled)

    result = replay_certificate(certificate_path)

    assert result["outcome"] == "SAFE_IN_FRAGMENT"


def test_detached_target_kernel_substitution_is_rejected(tmp_path):
    certificate_path, certificate = _packet(tmp_path)
    detached = tmp_path / "external" / "source" / "detached.py"
    detached.write_text(
        "@triton.jit\ndef renamed_kernel(dY_ptr, dX_ptr):\n    tl.store(dX_ptr, 0.0)\n",
        encoding="utf-8",
    )
    detached_bytes = detached.read_bytes()
    certificate["target"]["files"].append(
        {
            "packet_path": "external/source/detached.py",
            "target_path": "src/detached.py",
            "size": len(detached_bytes),
            "sha256": sha256_bytes(detached_bytes),
            "role": "source",
        }
    )
    certificate["target"]["kernels"][0]["file"] = "external/source/detached.py"
    certificate["digest"] = _sha({key: value for key, value in certificate.items() if key != "digest"})
    _write_canonical(certificate_path, certificate)

    with pytest.raises(ReplayError, match="source slice is not manifest-bound"):
        replay_certificate(certificate_path)


def test_arbitrary_extra_external_file_is_rejected(tmp_path):
    certificate_path, _ = _packet(tmp_path)
    extra = tmp_path / "external" / "extra.bin"
    extra.write_bytes(b"not declared")

    with pytest.raises(ReplayError, match="unexpected entry external/extra.bin"):
        replay_certificate(certificate_path)


def test_manifest_paths_are_distinct_and_casefolded_for_packet_identity(tmp_path):
    _, certificate = _packet(tmp_path)
    duplicate = copy.deepcopy(certificate["target"]["files"][0])
    duplicate["packet_path"] = "external/source/KERNEL.py"
    duplicate["target_path"] = "other/kernel.py"
    certificate["target"]["files"].append(duplicate)

    with pytest.raises(ReplayError, match="duplicate path"):
        _target_manifest(certificate["target"])


def test_manifest_rejects_ambiguous_kernel_binding_across_source_files(tmp_path):
    _, certificate = _packet(tmp_path)
    target = copy.deepcopy(certificate["target"])
    duplicate_file = copy.deepcopy(target["files"][0])
    duplicate_file["packet_path"] = "external/source/other.py"
    duplicate_file["target_path"] = "src/other.py"
    target["files"].append(duplicate_file)
    duplicate_kernel = copy.deepcopy(target["kernels"][0])
    duplicate_kernel["file"] = duplicate_file["packet_path"]
    target["kernels"].append(duplicate_kernel)

    with pytest.raises(ReplayError, match="duplicate bindings"):
        _target_manifest(target)


def test_manifest_accepts_distinct_kernel_source_hashes_across_source_files(tmp_path):
    _, certificate = _packet(tmp_path)
    target = copy.deepcopy(certificate["target"])
    duplicate_file = copy.deepcopy(target["files"][0])
    duplicate_file["packet_path"] = "external/source/other.py"
    duplicate_file["target_path"] = "src/other.py"
    target["files"].append(duplicate_file)
    distinct_kernel = copy.deepcopy(target["kernels"][0])
    distinct_kernel["file"] = duplicate_file["packet_path"]
    distinct_kernel["source_sha256"] = "b" * 64
    target["kernels"].append(distinct_kernel)

    assert len(_target_manifest(target)["kernels"]) == 2


def test_generic_packet_binding_accepts_a_normal_bounded_source(tmp_path):
    certificate_path, _ = _packet(tmp_path)

    assert replay_certificate(certificate_path)["valid"] is True


def test_generic_packet_binding_rejects_source_ast_over_the_existing_bound(tmp_path):
    source = "@triton.jit\ndef kernel(x_ptr):\n" + ("    value = 0\n" * 4097)
    encoded = source.encode()
    source_path = tmp_path / "external" / "source" / "kernel.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(encoded)
    source_sha256 = sha256_bytes(encoded)
    target = _target_manifest(
        {
            "repository": "https://example.invalid/repository",
            "revision": "a" * 40,
            "files": [
                {
                    "packet_path": "external/source/kernel.py",
                    "target_path": "src/kernel.py",
                    "size": len(encoded),
                    "sha256": source_sha256,
                    "role": "source",
                }
            ],
            "kernels": [
                {
                    "module": "synthetic.packet",
                    "qualname": "kernel",
                    "file": "external/source/kernel.py",
                    "source_sha256": source_sha256,
                }
            ],
            "specialization": {},
        }
    )
    capture = {
        "kernels": [
            {
                "module": "synthetic.packet",
                "qualname": "kernel",
                "source_sha256": source_sha256,
                "source": source,
            }
        ]
    }

    with pytest.raises(ReplayError, match="source slice is not manifest-bound") as error:
        _bind_capture_kernels(capture, target, tmp_path, "candidate")
    assert "target source AST exceeds its statement bound" in str(error.value.__cause__)


@pytest.mark.parametrize("bad_path", ["external/../kernel.py", "external\\source\\kernel.py", "external/source\x00.py"])
def test_manifest_rejects_unsafe_packet_paths(tmp_path, bad_path):
    _, certificate = _packet(tmp_path)
    certificate["target"]["files"][0]["packet_path"] = bad_path

    with pytest.raises(ReplayError, match="normalized relative path"):
        _target_manifest(certificate["target"])


def test_manifest_rejects_unknown_roles_and_source_aggregate_overrun(tmp_path):
    _, certificate = _packet(tmp_path)
    certificate["target"]["files"][0]["role"] = "operation"
    with pytest.raises(ReplayError, match="unknown role"):
        _target_manifest(certificate["target"])

    target = copy.deepcopy(certificate["target"])
    target["files"] = [
        {
            "packet_path": f"external/source/{index}.py",
            "target_path": f"src/{index}.py",
            "size": MAX_EXTERNAL_FILE_BYTES,
            "sha256": "a" * 64,
            "role": "source",
        }
        for index in range(3)
    ]
    target["kernels"][0]["file"] = "external/source/0.py"
    with pytest.raises(ReplayError, match="aggregate byte limit"):
        _target_manifest(target)


def test_capture_graph_requires_dense_topological_node_ids():
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    raw = candidate.to_dict()
    raw["forward"]["nodes"][1]["id"] = "node-1"

    with pytest.raises(ReplayError, match="dense topological n0..nN"):
        _capture(raw, "candidate")


def test_source_evidence_persists_memory_model_unknown_without_a_capture(tmp_path):
    source = """@triton.jit
def spinning_kernel(x_ptr, y_ptr):
    while tl.atomic_cas(x_ptr, 0, 1) != 0:
        pass
    tl.store(y_ptr, 1.0)
"""
    source_slice = _function_slice(source, "spinning_kernel")
    source_path = tmp_path / "external" / "source" / "tutorial.py"
    license_path = tmp_path / "external" / "source" / "LICENSE"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source.encode())
    license_path.write_bytes(b"MIT\n")
    evidence = {
        "schema": "daggertrace.aot-source-evidence/2",
        "module": "corpus.memory_model",
        "qualname": "spinning_kernel",
        "packet_path": "external/source/tutorial.py",
        "target_path": "tutorial.py",
        "file_sha256": sha256_bytes(source.encode()),
        "slice_sha256": sha256_bytes(source_slice.encode()),
        "source": source,
        "pointer_roles": ["x_ptr", "y_ptr"],
        "target": {
            "repository": "https://example.invalid/source",
            "revision": "a" * 40,
            "files": [
                {
                    "packet_path": "external/source/tutorial.py",
                    "target_path": "tutorial.py",
                    "size": len(source.encode()),
                    "sha256": sha256_bytes(source.encode()),
                    "role": "source",
                },
                {
                    "packet_path": "external/source/LICENSE",
                    "target_path": "LICENSE",
                    "size": 4,
                    "sha256": sha256_bytes(b"MIT\n"),
                    "role": "license",
                },
            ],
        },
        "analysis": {
            "outcome": "UNKNOWN",
            "reasons": ["MEMORY_MODEL"],
            "borrowed_writes": [],
            "collapse": None,
            "unknown_detail": UNKNOWN_MEMORY_MODEL_DETAIL,
        },
    }
    evidence_path = tmp_path / "source-evidence.json"
    _write_canonical(evidence_path, evidence)

    assert replay_source_evidence(evidence_path) == {
        "valid": True,
        "schema": "daggertrace.aot-source-evidence/2",
        "outcome": "UNKNOWN",
        "reasons": ["MEMORY_MODEL"],
        "source_sha256": evidence["file_sha256"],
    }


def test_source_evidence_rejects_detached_packet_source(tmp_path):
    source = """@triton.jit
def spinning_kernel(x_ptr, y_ptr):
    while tl.atomic_cas(x_ptr, 0, 1) != 0:
        pass
    tl.store(y_ptr, 1.0)
"""
    source_path = tmp_path / "external" / "source" / "tutorial.py"
    license_path = tmp_path / "external" / "source" / "LICENSE"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source.replace("1.0", "2.0"), encoding="utf-8")
    license_path.write_bytes(b"MIT\n")
    evidence = {
        "schema": "daggertrace.aot-source-evidence/2",
        "module": "corpus.memory_model",
        "qualname": "spinning_kernel",
        "packet_path": "external/source/tutorial.py",
        "target_path": "tutorial.py",
        "file_sha256": sha256_bytes(source.encode()),
        "slice_sha256": sha256_bytes(_function_slice(source, "spinning_kernel").encode()),
        "source": source,
        "pointer_roles": ["x_ptr", "y_ptr"],
        "target": {
            "repository": "https://example.invalid/source",
            "revision": "a" * 40,
            "files": [
                {
                    "packet_path": "external/source/tutorial.py",
                    "target_path": "tutorial.py",
                    "size": len(source.encode()),
                    "sha256": sha256_bytes(source.encode()),
                    "role": "source",
                },
                {
                    "packet_path": "external/source/LICENSE",
                    "target_path": "LICENSE",
                    "size": 4,
                    "sha256": sha256_bytes(b"MIT\n"),
                    "role": "license",
                },
            ],
        },
        "analysis": {
            "outcome": "UNKNOWN",
            "reasons": ["MEMORY_MODEL"],
            "borrowed_writes": [],
            "collapse": None,
            "unknown_detail": UNKNOWN_MEMORY_MODEL_DETAIL,
        },
    }
    evidence_path = tmp_path / "source-evidence.json"
    _write_canonical(evidence_path, evidence)

    with pytest.raises(ReplayError, match="target file|packet file differs"):
        replay_source_evidence(evidence_path)


def test_source_evidence_rejects_missing_packet_source(tmp_path):
    source = """@triton.jit
def spinning_kernel(x_ptr, y_ptr):
    while tl.atomic_cas(x_ptr, 0, 1) != 0:
        pass
    tl.store(y_ptr, 1.0)
"""
    license_path = tmp_path / "external" / "source" / "LICENSE"
    license_path.parent.mkdir(parents=True, exist_ok=True)
    license_path.write_bytes(b"MIT\n")
    evidence = {
        "schema": "daggertrace.aot-source-evidence/2",
        "module": "corpus.memory_model",
        "qualname": "spinning_kernel",
        "packet_path": "external/source/missing.py",
        "target_path": "tutorial.py",
        "file_sha256": sha256_bytes(source.encode()),
        "slice_sha256": sha256_bytes(_function_slice(source, "spinning_kernel").encode()),
        "source": source,
        "pointer_roles": ["x_ptr", "y_ptr"],
        "target": {
            "repository": "https://example.invalid/source",
            "revision": "a" * 40,
            "files": [
                {
                    "packet_path": "external/source/missing.py",
                    "target_path": "tutorial.py",
                    "size": len(source.encode()),
                    "sha256": sha256_bytes(source.encode()),
                    "role": "source",
                },
                {
                    "packet_path": "external/source/LICENSE",
                    "target_path": "LICENSE",
                    "size": 4,
                    "sha256": sha256_bytes(b"MIT\n"),
                    "role": "license",
                },
            ],
        },
        "analysis": {
            "outcome": "UNKNOWN",
            "reasons": ["MEMORY_MODEL"],
            "borrowed_writes": [],
            "collapse": None,
            "unknown_detail": UNKNOWN_MEMORY_MODEL_DETAIL,
        },
    }
    evidence_path = tmp_path / "source-evidence.json"
    _write_canonical(evidence_path, evidence)

    with pytest.raises(ReplayError, match="absent or unreadable|missing"):
        replay_source_evidence(evidence_path)


def test_function_binding_slices_raw_utf8_bytes_through_ast_end():
    source = """@first\r
@(\r
    second\r
)\r
def kernel(x):\r
    value = \"é\"\r
    return \"é\"\r
# outside the function\r
"""
    expected = source[: source.index("# outside")]

    assert _function_slice(source, "kernel") == expected


def test_function_binding_excludes_same_line_bytes_after_ast_end():
    source = "@triton.jit\ndef kernel(x): return x  # detached\n"
    expected = "@triton.jit\ndef kernel(x): return x"

    assert _function_slice(source, "kernel") == expected


def test_function_binding_leaves_lexically_valid_newer_python_syntax_opaque(monkeypatch):
    source = "type FutureAlias = int\n\n@triton.jit\ndef kernel(x):\n    return x\n"
    expected = "@triton.jit\ndef kernel(x):\n    return x\n"
    real_parse = ast.parse

    def parse(value, *args, **kwargs):
        if value == source:
            raise SyntaxError(
                "invalid syntax",
                ("<source>", 1, 1, "type FutureAlias = int\n"),
            )
        return real_parse(value, *args, **kwargs)

    monkeypatch.setattr(ast, "parse", parse)

    assert _function_slice(source, "kernel") == expected


def test_function_binding_counts_same_name_async_functions_as_conflicts():
    source = """@triton.jit
def kernel(x): return x
@triton.jit
async def kernel(x): return x
"""

    with pytest.raises(ValueError, match="not unique"):
        _function_slice(source, "kernel")


def test_function_binding_selects_suite_colon_after_lambda_return_annotation():
    source = """@triton.jit
def kernel(x) -> lambda: int:
    return x
"""

    assert _function_slice(source, "kernel") == source


def test_function_binding_preserves_lone_cr_bytes_during_tokenization():
    source = "@triton.jit\rdef kernel(x):\r    return x\r"

    result = _function_slice(source, "kernel")

    assert result.encode("utf-8") == source.encode("utf-8")


def test_function_binding_preserves_formfeed_decorator_prefix():
    source = "\f@triton.jit\ndef kernel(x):\n    return x\n"

    assert _function_slice(source, "kernel") == source


@pytest.mark.parametrize(
    ("suffix", "message"),
    [("]\n", "unmatched closing delimiter"), ("[)\n", "mismatched delimiter")],
)
def test_function_binding_rejects_invalid_delimiters_outside_bound_function(suffix, message):
    source = "@triton.jit\ndef kernel(x): return x\n" + suffix

    with pytest.raises(SyntaxError, match=message):
        _function_slice(source, "kernel")


def test_function_binding_rejects_error_tokens_outside_bound_function():
    source = "@triton.jit\ndef kernel(x): return x\nunterminated = \"\n"

    with pytest.raises(SyntaxError, match="invalid token|unterminated string literal"):
        _function_slice(source, "kernel")


@pytest.mark.parametrize(
    ("suffix", "message"),
    [
        ("@\n", "unfinished decorated function"),
        ("?\n", "invalid (?:token|operator)"),
    ],
)
def test_function_binding_rejects_invalid_lexical_suffixes(suffix, message):
    source = "@triton.jit\ndef kernel(x): return x\n" + suffix

    with pytest.raises(SyntaxError, match=message):
        _function_slice(source, "kernel")


@pytest.mark.parametrize(
    "suffix",
    ["+\n", "a=\n", "x=*\n"],
)
def test_function_binding_leaves_semantically_incomplete_unbound_suffixes_opaque(suffix):
    function = "@triton.jit\ndef kernel(x):\n    return x\n"

    assert _function_slice(function + suffix, "kernel") == function


@pytest.mark.parametrize("suffix", ["@decorator", "@decorator\nasync", "@decorator\ndef"])
def test_function_binding_rejects_unfinished_top_level_prefixes(suffix):
    source = "@triton.jit\ndef kernel(x): return x\n" + suffix

    with pytest.raises(SyntaxError):
        _function_slice(source, "kernel")


def test_closed_source_accepts_art_style_pure_operations_and_float():
    source = """@triton.jit
def topk_kernel(x_ptr, y_ptr, n_cols):
    offsets = tl.static_range(0, 4)
    values = tl.load(x_ptr + offsets)
    scaled = tl.where(values > 0, tl.exp(values), tl.max(values, axis=0))
    tl.store(y_ptr + offsets, float(scaled))
"""

    assert _source_effects(source, {"x_ptr", "y_ptr"}) == (
        ("x_ptr", "y_ptr", "n_cols"),
        ("x_ptr",),
        ("y_ptr",),
    )


def test_memory_model_source_is_an_explicit_unknown_not_safe():
    source = """@triton.jit
def spinning_kernel(x_ptr, y_ptr):
    while tl.atomic_cas(x_ptr, 0, 1) != 0:
        pass
    tl.store(y_ptr, 1.0)
"""

    with pytest.raises(ReplayError, match="UNKNOWN:MEMORY_MODEL"):
        _source_effects(source, {"x_ptr", "y_ptr"})


@pytest.mark.parametrize(
    "source",
    [
        """def pure_loop(pointer, output):
    while True:
        value = tl.load(pointer)
        tl.store(output, value)
        break
""",
        """def unknown_loop(pointer, output):
    while True:
        value = unknown(pointer)
        tl.store(output, value)
        break
""",
    ],
    ids=["pure", "unknown-call"],
)
def test_source_effects_classifies_non_protocol_while_as_source_profile(source):
    with pytest.raises(ReplayError, match="SOURCE_PROFILE") as error:
        _source_effects(source, {"pointer", "output"})

    assert "MEMORY_MODEL" not in str(error.value)


@pytest.mark.parametrize(
    "operation",
    [
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
    ],
)
def test_source_effects_classifies_every_memory_protocol_operation_as_unknown(operation):
    source = f"""def protocol(pointer, output):
    while True:
        tl.{operation}(pointer)
        break
    tl.store(output, 1.0)
"""

    with pytest.raises(ReplayError, match="UNKNOWN:MEMORY_MODEL"):
        _source_effects(source, {"pointer", "output"})


def test_generic_replay_source_has_no_example_target_identity():
    source = Path(__import__("daggertrace.replay", fromlist=["__file__"]).__file__).read_text(
        encoding="utf-8"
    ).lower()

    assert "liger" not in source
    assert "rms_norm" not in source
    assert "cuda:0" not in source
