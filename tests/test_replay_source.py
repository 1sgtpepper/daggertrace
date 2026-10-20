from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from daggertrace.replay import ReplayError, replay_certificate, replay_source_evidence
from daggertrace.replay_packet import (
    _filesystem_name,
    _packet_relative,
    _sha_bytes,
    _target_relative,
)
from daggertrace.replay_source import _source_effects
from daggertrace.replay_v2 import _validate_hops
from daggertrace.triton_protocols import UNKNOWN_MEMORY_MODEL_DETAIL
from daggertrace.validation_v2_fixtures import _micrograph
from _replay_fixtures import _write_canonical


def _write_source_packet(path: Path) -> Path:
    source = b"@triton.jit\ndef kernel(x_ptr):\n    while tl.atomic_cas(x_ptr, 0, 1) != 0:\n        pass\n    tl.store(x_ptr, 1.0)\n"
    source_path = path / "external" / "source" / "tree" / "kernel.py"
    license_path = path / "external" / "source" / "tree" / "LICENSE"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source)
    license_path.write_bytes(b"MIT\n")
    evidence = {
        "schema": "daggertrace.aot-source-evidence/2",
        "module": "source.kernel",
        "qualname": "kernel",
        "packet_path": "external/source/tree/kernel.py",
        "target_path": "src/kernel.py",
        "file_sha256": _sha_bytes(source),
        "slice_sha256": _sha_bytes(source),
        "source": source.decode("utf-8"),
        "pointer_roles": ["x_ptr"],
        "target": {
            "repository": "https://example.invalid/source",
            "revision": "a" * 40,
            "files": [
                {
                    "packet_path": "external/source/tree/kernel.py",
                    "target_path": "src/kernel.py",
                    "size": len(source),
                    "sha256": _sha_bytes(source),
                    "role": "source",
                },
                {
                    "packet_path": "external/source/tree/LICENSE",
                    "target_path": "LICENSE",
                    "size": 4,
                    "sha256": _sha_bytes(b"MIT\n"),
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
    evidence_path = path / "source-evidence.json"
    _write_canonical(evidence_path, evidence)
    return evidence_path


@pytest.mark.parametrize(
    "source",
    (
        """def kernel(tl, source, destination):
    tl.store(destination, tl.load(source))
""",
        """def kernel(source, destination):
    value = unknown_value.to(tl.float32)
    tl.store(destination, tl.load(source) + value)
""",
        """def kernel(source, destination):
    tl.store(destination, tl.load(source))
    return
""",
        """def kernel(source, destination=None):
    tl.store(destination, tl.load(source))
""",
    ),
)
def test_independent_source_replay_rejects_lexically_open_forms(source):
    with pytest.raises(ReplayError, match="SOURCE_PROFILE"):
        _source_effects(source, {"source", "destination"})


def test_independent_source_replay_admits_expand_dims_as_a_pure_shape_operation():
    source = """def kernel(source, destination):
    value = tl.expand_dims(tl.load(source), 0)
    tl.store(destination, value)
"""

    arguments, reads, writes = _source_effects(source, {"source", "destination"})

    assert arguments == ("source", "destination")
    assert reads == ("source",)
    assert writes == ("destination",)


def test_independent_source_replay_admits_base_2_math_and_joined_reduction():
    source = """def kernel(source, destination):
    value = tl.exp2(tl.load(source))
    normalizer = tl.log2(value)
    pairs = tl.join(value, value * normalizer)
    total, weighted = tl.split(tl.sum(pairs, axis=0))
    tl.store(destination, total + weighted)
"""

    arguments, reads, writes = _source_effects(source, {"source", "destination"})

    assert arguments == ("source", "destination")
    assert reads == ("source",)
    assert writes == ("destination",)


def test_source_evidence_replays_a_closed_target_file_manifest(tmp_path):
    evidence_path = _write_source_packet(tmp_path)

    result = replay_source_evidence(evidence_path)

    assert result["valid"] is True
    assert result["outcome"] == "UNKNOWN"
    assert result["source_sha256"] == _sha_bytes(
        (tmp_path / "external/source/tree/kernel.py").read_bytes()
    )


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX byte filesystem paths")
def test_source_evidence_replays_unicode_packet_paths_under_ascii_locale(tmp_path):
    packet = tmp_path / "packet"
    evidence_path = _write_source_packet(packet)
    source_path = packet / "external/source/tree/kernel.py"
    unicode_path = source_path.with_name("café.py")
    source_path.rename(unicode_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["packet_path"] = "external/source/tree/café.py"
    source_entry = next(item for item in evidence["target"]["files"] if item["role"] == "source")
    source_entry["packet_path"] = evidence["packet_path"]
    _write_canonical(evidence_path, evidence)

    script = "\n".join(
        (
            "import json",
            "from pathlib import Path",
            "from daggertrace.replay import replay_source_evidence",
            f"result = replay_source_evidence(Path({json.dumps(str(evidence_path), ensure_ascii=True)}))",
            "print(json.dumps(result, sort_keys=True))",
        )
    )
    environment = {
        **os.environ,
        "LC_ALL": "C",
        "PYTHONCOERCECLOCALE": "0",
        "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        "PYTHONUTF8": "0",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        env=environment,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["valid"] is True


def test_filesystem_name_decodes_only_valid_utf8_surrogate_escapes():
    assert _filesystem_name("café.py") == "café.py"
    assert _filesystem_name("caf\udcc3\udca9.py") == "café.py"
    assert _filesystem_name("\udcff.py") == "\udcff.py"


@pytest.mark.parametrize(
    "raw",
    (
        "/absolute",
        "//server/share",
        r"\\server\share",
        "foo//bar",
        "foo/./bar",
        "foo/../bar",
        "C:foo",
        "C:/foo",
        "foo/C:bar",
        "foo\x00bar",
        "\udcff",
    ),
)
@pytest.mark.parametrize(
    ("parser", "message"),
    (
        (_packet_relative, "normalized relative path"),
        (_target_relative, "valid provenance"),
    ),
)
def test_relative_path_grammar_rejects_noncanonical_and_windows_forms(raw, parser, message):
    with pytest.raises(ReplayError, match=message):
        parser(raw, "path")


def test_relative_path_grammar_accepts_canonical_unicode_paths():
    assert _packet_relative("external/source/café.py", "packet") == Path(
        "external/source/café.py"
    )
    assert _target_relative("src/café.py", "target") == "src/café.py"


def test_source_evidence_rejects_an_oversized_ast_before_effect_replay(tmp_path):
    evidence_path = _write_source_packet(tmp_path)
    source = (
        "@triton.jit\ndef kernel(x_ptr):\n"
        + "".join(f"    value_{index} = {index}\n" for index in range(4_100))
        + "    while tl.atomic_cas(x_ptr, 0, 1) != 0:\n"
        + "        pass\n"
    )
    encoded = source.encode("utf-8")
    source_path = tmp_path / "external/source/tree/kernel.py"
    source_path.write_bytes(encoded)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["source"] = source
    evidence["file_sha256"] = _sha_bytes(encoded)
    evidence["slice_sha256"] = _sha_bytes(encoded)
    source_entry = next(
        item for item in evidence["target"]["files"] if item["role"] == "source"
    )
    source_entry.update(size=len(encoded), sha256=_sha_bytes(encoded))
    _write_canonical(evidence_path, evidence)

    with pytest.raises(ReplayError, match="statement bound"):
        replay_source_evidence(evidence_path)


@pytest.mark.parametrize("mutation", ["license", "extra", "missing", "symlink"])
def test_source_evidence_rejects_packet_inventory_mutations(tmp_path, mutation):
    evidence_path = _write_source_packet(tmp_path)
    license_path = tmp_path / "external/source/tree/LICENSE"
    if mutation == "license":
        license_path.write_bytes(b"substitute\n")
    elif mutation == "extra":
        (tmp_path / "external/source/tree/extra.py").write_bytes(b"extra\n")
    elif mutation == "missing":
        license_path.unlink()
    else:
        license_path.unlink()
        license_path.symlink_to(tmp_path / "host-license")
        (tmp_path / "host-license").write_bytes(b"MIT\n")

    with pytest.raises(ReplayError, match="packet inventory|target file"):
        replay_source_evidence(evidence_path)


@pytest.mark.parametrize(
    "edit",
    [
        lambda node: node["args"]["tuple"].append("extra"),
        lambda node: node.update(kwargs={"unexpected": True}),
    ],
)
def test_forward_hop_getitem_requires_exact_args_and_empty_kwargs(edit):
    source = """def kernel(X_ptr, Y_ptr):
    value = tl.load(X_ptr)
    tl.store(Y_ptr, value)
"""
    kernel = {
        "id": "0" * 64,
        "source": source,
        "argument_names": ["X_ptr", "Y_ptr"],
        "reads": ["X_ptr"],
        "writes": ["Y_ptr"],
    }
    tensor = {
        "shape": [1],
        "stride": [1],
        "dtype": "float32",
        "device": "cpu",
        "storage_offset": 0,
        "alias_group": "input",
    }
    output = {
        "shape": [1],
        "stride": [1],
        "dtype": "float32",
        "device": "cpu",
        "storage_offset": 0,
        "alias_group": "output",
    }
    graph = {
        "nodes": [
            {"id": "n0", "target": "placeholder", "tensor": tensor},
            {"id": "n1", "target": "aten.empty", "tensor": output},
            {
                "id": "n2",
                "target": "triton.functional",
                "kwargs": {
                    "kernel": kernel["id"],
                    "bindings": {"X_ptr": {"ref": "n0"}, "Y_ptr": {"ref": "n1"}},
                    "writes": ["Y_ptr"],
                    "constants": {},
                    "grid": [{"tuple": [1, 1, 1]}],
                },
            },
            {
                "id": "n3",
                "target": "operator.getitem",
                "args": {"tuple": [{"ref": "n2"}, "Y_ptr"]},
                "kwargs": {},
                "tensor": output,
            },
        ]
    }
    capture = {"kernels": [kernel], "forward": graph, "backward": {"nodes": []}}
    _validate_hops(capture, "unit")

    mutant = copy.deepcopy(capture)
    edit(mutant["forward"]["nodes"][-1])
    with pytest.raises(ReplayError, match="exact getitem grammar"):
        _validate_hops(mutant, "unit")


def test_generic_replay_rejects_encoded_tensor_ref_in_hop_constants():
    source = """def kernel(X_ptr, Y_ptr, BLOCK_SIZE):
    value = tl.load(X_ptr)
    tl.store(Y_ptr, value)
"""
    kernel = {
        "id": "0" * 64,
        "source": source,
        "argument_names": ["X_ptr", "Y_ptr", "BLOCK_SIZE"],
        "reads": ["X_ptr"],
        "writes": ["Y_ptr"],
    }
    tensor = {"alias_group": "input"}
    output = {"alias_group": "output"}
    capture = {
        "kernels": [kernel],
        "forward": {"nodes": []},
        "backward": {
            "nodes": [
                {
                    "id": "n0",
                    "target": "placeholder",
                    "tensor": tensor,
                },
                {
                    "id": "n1",
                    "target": "aten.empty",
                    "tensor": output,
                },
                {
                    "id": "n2",
                    "target": "triton.functional",
                    "kwargs": {
                        "kernel": kernel["id"],
                        "bindings": {"X_ptr": {"ref": "n0"}, "Y_ptr": {"ref": "n1"}},
                        "writes": ["Y_ptr"],
                        "constants": {"BLOCK_SIZE": {"ref": "n0"}},
                        "grid": [{"tuple": [1, 1, 1]}],
                    },
                },
            ],
        },
    }

    with pytest.raises(ReplayError, match="constants contain encoded tensor refs"):
        _validate_hops(capture, "unit")


@pytest.mark.parametrize(
    "grid",
    [
        "not-a-grid",
        [],
        [{"tuple": [1, 1]}],
        [{"tuple": [1, 0, 1]}],
        [{"tuple": [True, 1, 1]}],
    ],
)
def test_generic_replay_rejects_malformed_hop_grid_structure(grid):
    capture, _control = _micrograph(False, "direct", False, 1, False)
    raw = capture.to_dict()
    hop = next(node for node in raw["backward"]["nodes"] if node["target"] == "triton.functional")
    hop["kwargs"]["grid"] = grid

    with pytest.raises(ReplayError, match="grid"):
        _validate_hops(raw, "candidate")


def test_resealed_source_profile_rejects_unknown_non_tl_call():
    source = """def kernel(dY_ptr, dX_ptr):
    hidden = evil(dY_ptr)
    value = tl.load(dY_ptr)
    tl.store(dX_ptr, value + hidden)
"""
    with pytest.raises(ReplayError, match="call evil is outside the supported source subset"):
        _source_effects(source, {"dY_ptr", "dX_ptr"})


def test_resealed_source_profile_rejects_to_on_a_local_bound_from_global_state():
    source = """def kernel(dY_ptr, dX_ptr):
    value = unknown_value
    converted = value.to(tl.float32)
    tl.store(dX_ptr, tl.load(dY_ptr) + converted)
"""
    with pytest.raises(ReplayError, match=".to receiver is not a supported Triton expression"):
        _source_effects(source, {"dY_ptr", "dX_ptr"})


def test_checked_art_packet_replays_when_unbound_source_uses_newer_python_syntax():
    root = Path(__file__).resolve().parents[1]
    result = replay_certificate(
        root / "evidence/results/bundle/art/openpipe-art-local-stats/certificate.json"
    )

    assert result["valid"] is True
    assert result["outcome"] == "SAFE_IN_FRAGMENT"
    assert result["reasons"] == ["FRESH_DESTINATION"]
    assert result["certificate_digest"] == (
        "48080d87ab85a0a5c72e168b8bb4bf426926fd7c8702160af599343eb8fdc838"
    )
