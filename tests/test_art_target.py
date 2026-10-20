from __future__ import annotations

import builtins
import copy
import io
import json
import shutil
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from daggertrace import (
    art_capture,
    art_target,
    cli_replay,
    liger_target,
    ownership as _ownership,
    source_files,
    target_support,
    targets,
)
from daggertrace.cli import replay_certificate
from daggertrace.canonical import sha256_json, write_canonical_json
from daggertrace.effects import profile_triton_source
from daggertrace.errors import InvalidArtifact, TargetError, UnsupportedCapture
from daggertrace.model import Capture, ref
from daggertrace.replay import ReplayError
from daggertrace.replay_packet import _runtime_evidence as replay_runtime_evidence
from daggertrace.validation_v2_fixtures import _micrograph
from daggertrace.version import CAPTURE_SCHEMA, CERTIFICATE_SCHEMA


class _Response:
    def __init__(self, payload, headers=None):
        self.payload = payload
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def read(self, _limit):
        return self.payload


@pytest.mark.parametrize(
    ("module_name", "invoke", "message"),
    (
        (
            "triton.runtime.jit",
            lambda: art_capture._validate_art_kernel_objects(object()),
            "ART adapter requires Triton JIT support",
        ),
        (
            "torch._inductor",
            lambda: art_target._compile_art_lane(None, None, None),
            "PyTorch Inductor is unavailable",
        ),
    ),
)
def test_art_import_guards_normalize_native_loader_oserror(
    monkeypatch, module_name, invoke, message
):
    failure = OSError("native loader failed")
    real_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == module_name:
            raise failure
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(art_target, "_require_module", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(TargetError, match=message) as raised:
        invoke()

    assert raised.value.__cause__ is failure


def _exact_tree_fixture(monkeypatch, tmp_path):
    root = tmp_path / "source" / "ART-tree"
    expected = {
        art_target.ART_SOURCE_PATH: art_target.ART_SOURCE_SHA256,
        art_target.ART_LICENSE_PATH: art_target.ART_LICENSE_SHA256,
        art_target.ART_NOTICE_PATH: art_target.ART_NOTICE_SHA256,
        art_target.ART_GPL_PATH: art_target.ART_GPL_SHA256,
        art_target.ART_LGPL_PATH: art_target.ART_LGPL_SHA256,
    }
    for relative in expected:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))

    real_sha256_bytes = source_files.sha256_bytes

    def exact_digest(value):
        try:
            relative = value.decode("utf-8")
        except UnicodeDecodeError:
            return real_sha256_bytes(value)
        return expected.get(relative, real_sha256_bytes(value))

    monkeypatch.setattr(source_files, "sha256_bytes", exact_digest)
    return root, expected


def _art_candidate():
    candidate, _control = _micrograph(False, "direct", False, 1, False)
    base_kernel = candidate.kernels[0]

    def make_kernel(index, qualname):
        runtime = copy.deepcopy(base_kernel.runtime)
        runtime["config"]["num_warps"] = 8
        runtime["jit_cache_key"] = f"{index + 1:064x}"
        identity = {
            "module": "art.trainer_rank.topk",
            "qualname": qualname,
            "source_sha256": base_kernel.source_sha256,
            "arguments": list(base_kernel.argument_names),
            "reads": list(base_kernel.reads),
            "writes": list(base_kernel.writes),
            "runtime": runtime,
        }
        return replace(
            base_kernel,
            id=sha256_json(identity),
            module="art.trainer_rank.topk",
            qualname=qualname,
            runtime=runtime,
        )

    kernels = tuple(make_kernel(index, name) for index, name in enumerate(art_target._ART_KERNEL_NAMES))
    hop_template = next(
        node for node in candidate.backward.nodes if node.target == "triton.functional"
    )
    placeholder = candidate.forward.nodes[0]
    output = candidate.forward.nodes[-1]
    allocation_template = next(
        node for node in candidate.backward.nodes if node.target == "aten.zeros_like"
    )

    def allocation(node_id, alias_group):
        return replace(
            allocation_template,
            id=node_id,
            target="aten.empty_like",
            args={"tuple": [ref("n0")]},
            tensor=replace(allocation_template.tensor, alias_group=alias_group),
        )

    def hop(node_id, kernel_id, qualname, destination):
        kernel = next(kernel for kernel in kernels if kernel.id == kernel_id)
        return replace(
            hop_template,
            id=node_id,
            kwargs={
                **hop_template.kwargs,
                "kernel": kernel_id,
                "bindings": {
                    kernel.reads[0]: ref("n0"),
                    kernel.writes[0]: ref(destination),
                },
                "grid": art_capture._expected_art_grid(qualname),
            },
        )

    forward = replace(
        candidate.forward,
        nodes=(
            placeholder,
            allocation("n1", "forward-destination-1"),
            hop("n2", kernels[0].id, art_target._ART_KERNEL_NAMES[0], "n1"),
            allocation("n3", "forward-destination-2"),
            hop("n4", kernels[1].id, art_target._ART_KERNEL_NAMES[1], "n3"),
            replace(output, id="n5", args={"tuple": [{"tuple": [ref("n0")]}]}),
        ),
    )

    def edit_backward_node(node):
        if node.target == "aten.zeros_like":
            return replace(node, target="aten.empty_like")
        if node.target == "triton.functional":
            return replace(
                node,
                kwargs={
                    **node.kwargs,
                    "kernel": kernels[2].id,
                    "grid": art_capture._expected_art_grid(art_target._ART_KERNEL_NAMES[2]),
                },
            )
        return node

    backward = replace(
        candidate.backward,
        nodes=tuple(edit_backward_node(node) for node in candidate.backward.nodes),
    )
    return replace(
        candidate,
        forward=forward,
        backward=backward,
        kernels=kernels,
        diagnostics={
            "callbacks": {"forward": 1, "backward": 1},
            "aot_autograd": {"ok": 1, "total": 1},
            "inputs": [
                {
                    "shape": [2, 8192],
                    "stride": [8192, 1],
                    "storage_offset": 0,
                    "dtype": "float32",
                    "device": "cuda:0",
                    "requires_grad": True,
                }
            ],
            "grad_output": {
                "shape": [2],
                "stride": [1],
                "storage_offset": 0,
                "dtype": "float32",
                "device": "cuda:0",
                "requires_grad": False,
            },
        },
    )


def _recorded_art_capture():
    path = Path(__file__).parents[1] / "evidence" / "results" / "bundle" / "art" / "openpipe-art-local-stats" / "candidate.capture.json"
    return json.loads(path.read_text())


def _art_capture_device(raw, device):
    for value in (*raw["diagnostics"]["inputs"], raw["diagnostics"]["grad_output"]):
        value["device"] = device
    for graph in (raw["forward"], raw["backward"]):
        for node in graph["nodes"]:
            if node["tensor"] is not None:
                node["tensor"]["device"] = device
            if node["kwargs"].get("device") is not None:
                node["kwargs"]["device"]["torch"] = device


def test_art_capture_specialization_accepts_the_fixed_target():
    art_target._validate_art_capture_specialization(Capture.from_dict(_recorded_art_capture()))


def test_art_capture_specialization_accepts_consistent_cuda_one():
    raw = _recorded_art_capture()
    _art_capture_device(raw, "cuda:1")
    art_target._validate_art_capture_specialization(Capture.from_dict(raw))


@pytest.mark.parametrize(
    "mutate",
    (
        lambda raw: raw["diagnostics"]["inputs"][0].update(shape=[1, 8192]),
        lambda raw: next(
            node for node in raw["forward"]["nodes"]
            if node["target"] == "triton.functional"
            and node["kwargs"]["constants"].get("vocab_size") == 8192
        )["kwargs"]["constants"].update(vocab_size=1),
        lambda raw: next(node for node in raw["forward"]["nodes"] if node["target"] == "aten.empty").update(target="aten.zeros"),
        lambda raw: next(
            node for node in raw["forward"]["nodes"] if node["id"] == "n1"
        )["kwargs"]["dtype"].update(torch="torch.int64"),
        lambda raw: next(
            node for node in raw["forward"]["nodes"] if node["id"] == "n1"
        )["kwargs"].update(pin_memory=True),
        lambda raw: next(node for node in raw["forward"]["nodes"] if node["kwargs"].get("constants", {}).get("vocab_size") == 8192)["kwargs"]["bindings"].update(partial_values_ptr={"ref": "n1"}),
        lambda raw: raw["forward"]["nodes"][-1]["args"]["tuple"][0]["tuple"][0].update(ref="n15"),
    ),
)
def test_art_capture_specialization_rejects_a_resealed_mutation(mutate):
    raw = _recorded_art_capture()
    mutate(raw)

    with pytest.raises(UnsupportedCapture):
        art_target._validate_art_capture_specialization(Capture.from_dict(raw))


def _art_observations(candidate):
    return [
        {
            **kernel.runtime,
            "hooks": {
                "reset_to_zero": [],
                "restore_value": [],
                "user_defined_pre_hook": False,
                "user_defined_post_hook": False,
            },
        }
        for kernel in candidate.kernels
    ]


def test_target_registry_dispatches_both_adapters(monkeypatch, tmp_path):
    calls = []

    def fake_liger(output_dir, **kwargs):
        calls.append(("liger", output_dir, kwargs))
        return {"target": "liger"}

    def fake_art(output_dir, **kwargs):
        calls.append(("art", output_dir, kwargs))
        return {"target": "art"}

    monkeypatch.setattr(liger_target, "audit_liger", fake_liger)
    monkeypatch.setattr(art_target, "audit_art", fake_art)
    assert targets._audit_target("liger-rmsnorm", tmp_path / "liger") == {"target": "liger"}
    assert targets._audit_target("openpipe-art-local-stats", tmp_path / "art") == {"target": "art"}
    assert [call[0] for call in calls] == ["liger", "art"]

    with pytest.raises(TargetError, match="unknown target"):
        targets._audit_target("not-a-target", tmp_path)


def test_art_tree_requires_selected_files_to_be_regular(monkeypatch, tmp_path):
    root, _expected = _exact_tree_fixture(monkeypatch, tmp_path)
    source = root / art_target.ART_SOURCE_PATH
    source.unlink()
    source.symlink_to(root / art_target.ART_LICENSE_PATH)

    with pytest.raises(TargetError, match="not a regular file"):
        art_target._validate_art_tree(root)


def test_art_loader_uses_the_neutral_exact_source_owner(monkeypatch, tmp_path):
    root, _expected = _exact_tree_fixture(monkeypatch, tmp_path)
    expected_module = SimpleNamespace()
    calls = []
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "art" or name.startswith("art.")
    }

    def load(name, path, expected_sha256, target_name):
        calls.append((name, path, expected_sha256, target_name))
        return expected_module

    monkeypatch.setattr(art_capture, "_load_exact_module", load)
    monkeypatch.setattr(art_capture, "_validate_art_kernel_objects", lambda _module: None)
    try:
        module = art_capture._load_art_module(root)
    finally:
        for name in tuple(sys.modules):
            if name == "art" or name.startswith("art."):
                del sys.modules[name]
        sys.modules.update(previous)

    assert module is expected_module
    assert calls == [
        (
            "art.trainer_rank.topk",
            root / art_target.ART_SOURCE_PATH,
            art_target.ART_SOURCE_SHA256,
            "ART",
        )
    ]


def test_art_loader_cleans_namespace_after_kernel_validation_failure(monkeypatch, tmp_path):
    root, _expected = _exact_tree_fixture(monkeypatch, tmp_path)
    failure = TargetError("invalid ART kernel bindings")
    cause = RuntimeError("kernel metadata is incomplete")
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "art" or name.startswith("art.")
    }

    monkeypatch.setattr(art_capture, "_load_exact_module", lambda *_args: SimpleNamespace())

    def validate(_module):
        raise failure from cause

    monkeypatch.setattr(art_capture, "_validate_art_kernel_objects", validate)
    try:
        with pytest.raises(TargetError) as raised:
            art_capture._load_art_module(root)
        assert raised.value is failure
        assert raised.value.__cause__ is cause
        assert not any(name == "art" or name.startswith("art.") for name in sys.modules)
    finally:
        for name in tuple(sys.modules):
            if name == "art" or name.startswith("art."):
                del sys.modules[name]
        sys.modules.update(previous)


def test_art_capture_context_restores_the_runtime_observer(monkeypatch):
    original_runtime = object()
    monkeypatch.setattr(art_capture._capture_module, "_kernel_runtime", original_runtime)

    with art_capture._art_capture_context() as observations:
        assert observations == []
        assert callable(art_capture._capture_module._kernel_runtime)

    assert art_capture._capture_module._kernel_runtime is original_runtime


def test_art_tree_requires_exact_selected_file_hash(monkeypatch, tmp_path):
    root, expected = _exact_tree_fixture(monkeypatch, tmp_path)
    expected[art_target.ART_SOURCE_PATH] = "0" * 64

    with pytest.raises(TargetError, match="hash mismatch"):
        art_target._validate_art_tree(root)


def test_art_tree_rejects_symlinked_source_directory(monkeypatch, tmp_path):
    root, _expected = _exact_tree_fixture(monkeypatch, tmp_path)
    source_directory = root / "src" / "art" / "trainer_rank"
    shutil.rmtree(source_directory)
    outside = tmp_path / "outside"
    outside.mkdir()
    source_directory.symlink_to(outside, target_is_directory=True)

    with pytest.raises(TargetError, match="path component is not a directory"):
        art_target._validate_art_tree(root)


def test_art_tree_rejects_symlinked_root(monkeypatch, tmp_path):
    root, _expected = _exact_tree_fixture(monkeypatch, tmp_path)
    alias = tmp_path / "target-alias"
    alias.symlink_to(root, target_is_directory=True)

    with pytest.raises(TargetError, match="tree is not a directory"):
        art_target._validate_art_tree(alias)


def test_art_identity_and_license_staging_are_exact(monkeypatch, tmp_path):
    root, expected = _exact_tree_fixture(monkeypatch, tmp_path)
    output = tmp_path / "packet"
    art_target._stage_art_sources(root, output)
    identity = art_target._target_identity(root, None)

    assert identity["repository"] == art_target.ART_REPOSITORY
    assert identity["revision"] == art_target.ART_REVISION
    assert set(identity) == {"repository", "revision", "files", "kernels", "specialization"}
    assert identity["kernels"] == [
        {
            "file": "external/openpipe-art/tree/src/art/trainer_rank/topk.py",
            "module": "art.trainer_rank.topk",
            "qualname": "_stats_backward_kernel",
            "source_sha256": art_target._ART_KERNEL_SOURCE_SHA256["_stats_backward_kernel"],
        },
        {
            "file": "external/openpipe-art/tree/src/art/trainer_rank/topk.py",
            "module": "art.trainer_rank.topk",
            "qualname": "_stats_stage1_kernel",
            "source_sha256": art_target._ART_KERNEL_SOURCE_SHA256["_stats_stage1_kernel"],
        },
        {
            "file": "external/openpipe-art/tree/src/art/trainer_rank/topk.py",
            "module": "art.trainer_rank.topk",
            "qualname": "_stats_stage2_kernel",
            "source_sha256": art_target._ART_KERNEL_SOURCE_SHA256["_stats_stage2_kernel"],
        },
    ]
    assert identity["files"] == [
        {
            "packet_path": "external/openpipe-art/tree/src/art/trainer_rank/topk.py",
            "target_path": art_target.ART_SOURCE_PATH,
            "size": len((root / art_target.ART_SOURCE_PATH).read_bytes()),
            "sha256": art_target.ART_SOURCE_SHA256,
            "role": "source",
        },
        {
            "packet_path": "external/openpipe-art/tree/LICENSE",
            "target_path": art_target.ART_LICENSE_PATH,
            "size": len((root / art_target.ART_LICENSE_PATH).read_bytes()),
            "sha256": art_target.ART_LICENSE_SHA256,
            "role": "license",
        },
        {
            "packet_path": "external/openpipe-art/tree/THIRD-PARTY-NOTICES",
            "target_path": art_target.ART_NOTICE_PATH,
            "size": len((root / art_target.ART_NOTICE_PATH).read_bytes()),
            "sha256": art_target.ART_NOTICE_SHA256,
            "role": "notice",
        },
        {
            "packet_path": "external/openpipe-art/tree/licenses/GPL-3.0.txt",
            "target_path": art_target.ART_GPL_PATH,
            "size": len((root / art_target.ART_GPL_PATH).read_bytes()),
            "sha256": art_target.ART_GPL_SHA256,
            "role": "license",
        },
        {
            "packet_path": "external/openpipe-art/tree/licenses/LGPL-3.0.txt",
            "target_path": art_target.ART_LGPL_PATH,
            "size": len((root / art_target.ART_LGPL_PATH).read_bytes()),
            "sha256": art_target.ART_LGPL_SHA256,
            "role": "license",
        },
    ]
    assert identity["specialization"] == {
        "shape": [2, 8192],
        "dtype": "float32",
        "device": "cuda",
        "k": 2,
        "block_v": 4096,
        "expression": "local_sum",
        "row_max": "detached",
        "source_launches": {
            "decorator": "triton.jit",
            "num_warps": 8,
            "kernel_count": 3,
        },
        "runtime_capture": None,
    }
    staged = output / "external" / "openpipe-art" / "tree"
    for relative in expected:
        assert (staged / relative).read_bytes() == (root / relative).read_bytes()


def test_art_identity_is_packet_relative_even_without_a_capture(monkeypatch, tmp_path):
    root, _expected = _exact_tree_fixture(monkeypatch, tmp_path)
    identity = art_target._target_identity(root, None)

    assert identity["kernels"]
    assert all(
        path["packet_path"].startswith("external/openpipe-art/tree/")
        for path in identity["files"]
    )
    assert all(
        binding["file"] == "external/openpipe-art/tree/src/art/trainer_rank/topk.py"
        for binding in identity["kernels"]
    )


def test_art_static_kernel_hashes_match_checked_capture_and_certificate_metadata():
    packet = (
        Path(__file__).resolve().parents[1]
        / "evidence/results/bundle/art/openpipe-art-local-stats"
    )
    capture_kernels = json.loads((packet / "candidate.capture.json").read_text())["kernels"]
    certificate_kernels = json.loads((packet / "certificate.json").read_text())["target"]["kernels"]

    capture_hashes = {
        (kernel["module"], kernel["qualname"]): kernel["source_sha256"]
        for kernel in capture_kernels
    }
    certificate_hashes = {
        (kernel["module"], kernel["qualname"]): kernel["source_sha256"]
        for kernel in certificate_kernels
    }
    static_hashes = {
        (art_target._ART_KERNEL_MODULE, qualname): digest
        for qualname, digest in art_target._ART_KERNEL_SOURCE_SHA256.items()
    }

    assert len(capture_kernels) == len(art_target._ART_KERNEL_NAMES)
    assert len(certificate_kernels) == len(art_target._ART_KERNEL_NAMES)
    assert capture_hashes == certificate_hashes == static_hashes


def test_art_identity_binds_three_runtime_kernels_and_per_hop_grids(monkeypatch, tmp_path):
    root, _expected = _exact_tree_fixture(monkeypatch, tmp_path)
    candidate = _art_candidate()
    observations = _art_observations(candidate)
    identity = art_target._target_identity(root, candidate, observations)

    assert set(identity) == {"repository", "revision", "files", "kernels", "specialization"}
    assert {
        (binding["module"], binding["qualname"])
        for binding in identity["kernels"]
    } == {
        ("art.trainer_rank.topk", name) for name in art_target._ART_KERNEL_NAMES
    }
    assert all(set(binding) == {"module", "qualname", "file", "source_sha256"} for binding in identity["kernels"])
    assert {binding["file"] for binding in identity["kernels"]} == {
        "external/openpipe-art/tree/src/art/trainer_rank/topk.py"
    }
    assert {binding["source_sha256"] for binding in identity["kernels"]} == {
        kernel.source_sha256 for kernel in candidate.kernels
    }
    observations = _art_observations(candidate)
    assert len(observations) == 3
    assert all(
        observation["wrapper_type"] == "triton.runtime.autotuner.Autotuner"
        and observation["leaf_type"] == "triton.runtime.jit.JITFunction"
        and observation["config"]["num_warps"] == 8
        and observation["hooks"] == {
            "reset_to_zero": [],
            "restore_value": [],
            "user_defined_pre_hook": False,
            "user_defined_post_hook": False,
        }
        for observation in observations
    )
    expected_grids = [
        {"graph": "forward", "qualname": "_stats_stage1_kernel", "grid": [{"tuple": [2, 2, 1]}]},
        {"graph": "forward", "qualname": "_stats_stage2_kernel", "grid": [{"tuple": [2, 1, 1]}]},
        {"graph": "backward", "qualname": "_stats_backward_kernel", "grid": [{"tuple": [2, 2, 1]}]},
    ]
    assert identity["specialization"]["runtime_capture"]["observations"] == observations
    assert identity["specialization"]["runtime_capture"]["grids"] == expected_grids


def test_art_runtime_binding_rejects_source_only_jit_observation():
    candidate = _art_candidate()
    observations = _art_observations(candidate)
    observations[0]["wrapper_type"] = "triton.runtime.jit.JITFunction"
    with pytest.raises(UnsupportedCapture, match="wrapper observation"):
        art_target._validate_art_runtime_bindings(candidate, observations)


def test_art_grid_derivation_tracks_the_declared_specialization(monkeypatch):
    assert art_capture._expected_art_grid("_stats_stage1_kernel") == [{"tuple": [2, 2, 1]}]
    assert art_capture._expected_art_grid("_stats_stage2_kernel") == [{"tuple": [2, 1, 1]}]
    assert art_capture._expected_art_grid("_stats_backward_kernel") == [{"tuple": [2, 2, 1]}]

    monkeypatch.setattr(art_target, "ART_SHAPE", (1, 1024))
    assert art_capture._expected_art_grid("_stats_stage1_kernel") == [{"tuple": [1, 1, 1]}]
    assert art_capture._expected_art_grid("_stats_stage2_kernel") == [{"tuple": [1, 1, 1]}]
    assert art_capture._expected_art_grid("_stats_backward_kernel") == [{"tuple": [1, 1, 1]}]


def test_art_capture_rejects_a_grid_bound_to_the_wrong_kernel():
    candidate = _art_candidate()
    forward_nodes = list(candidate.forward.nodes)
    hop = forward_nodes[2]
    forward_nodes[2] = replace(
        hop,
        kwargs={**hop.kwargs, "grid": [{"tuple": [2, 1, 1]}]},
    )
    malformed = replace(candidate, forward=replace(candidate.forward, nodes=tuple(forward_nodes)))

    with pytest.raises(UnsupportedCapture, match="HOP grid"):
        art_target._validate_art_runtime_bindings(malformed, _art_observations(candidate))


def test_fetch_art_cache_hit_revalidates_without_network(monkeypatch, tmp_path):
    root, _expected = _exact_tree_fixture(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    cached = cache / f"ART-{art_target.ART_REVISION}"
    cached.parent.mkdir()
    root.rename(cached)
    monkeypatch.setattr(
        target_support.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network reached")),
    )

    assert art_target.fetch_art(cache) == cached


def test_fetch_art_rejects_archive_path_escape(monkeypatch, tmp_path):
    payload = io.BytesIO()
    prefix = f"ART-{art_target.ART_REVISION}/"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(f"{prefix}../../escape", b"escape")
    monkeypatch.setattr(
        target_support.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(payload.getvalue()),
    )

    with pytest.raises(TargetError, match="unsafe ART archive member"):
        art_target.fetch_art(tmp_path / "cache")


def test_fetch_art_enforces_download_bound_before_extraction(monkeypatch, tmp_path):
    monkeypatch.setattr(target_support, "_MAX_ARCHIVE_BYTES", 4)
    monkeypatch.setattr(
        target_support.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(b"12345"),
    )

    with pytest.raises(TargetError, match="download limit"):
        art_target.fetch_art(tmp_path / "cache")


def test_art_source_profile_admits_required_pure_calls_and_bare_float():
    source = """
    @triton.jit
    def kernel(x_ptr, y_ptr):
        value = tl.load(x_ptr)
        row_max = tl.max(value, axis=0)
        value = tl.exp(value - row_max)
        for _ in tl.static_range(0, 1):
            value = tl.where(value > float("-inf"), value, 0.0)
        tl.store(y_ptr, value)
    """

    effects = profile_triton_source(source)
    assert effects.arguments == ("x_ptr", "y_ptr")
    assert effects.reads == ("x_ptr",)
    assert effects.writes == ("y_ptr",)
    assert {event.operation for event in effects.events} == {"tl.load", "tl.store"}


def test_art_source_profile_rejects_unlisted_triton_calls():
    source = """
    @triton.jit
    def kernel(x_ptr, y_ptr):
        value = tl.sqrt(tl.load(x_ptr))
        tl.store(y_ptr, value)
    """

    with pytest.raises(UnsupportedCapture, match="tl.sqrt"):
        profile_triton_source(source)


@pytest.mark.parametrize("num_warps", [4, 0, 16])
def test_art_runtime_config_rejects_noncanonical_observed_config(num_warps):
    config = {
        "kwargs": {},
        "num_warps": num_warps,
        "num_ctas": 1,
        "num_stages": 3,
        "maxnreg": None,
        "pre_hook": None,
        "ir_override": None,
    }

    with pytest.raises(InvalidArtifact, match="exact ART runtime profile"):
        art_capture._validate_art_runtime_config(config)


def test_art_runtime_config_accepts_the_observed_hop_config():
    art_capture._validate_art_runtime_config(
        {
            "kwargs": {},
            "num_warps": 8,
            "num_ctas": 1,
            "num_stages": 3,
            "maxnreg": None,
            "pre_hook": None,
            "ir_override": None,
        }
    )


def _art_runtime_fixture():
    delta = {"output_max_abs": 0.0, "gx_max_abs": 0.0, "gw_max_abs": 0.0}
    occurrence = {
        "generated_kernel_count": 1,
        "ir_nodes_pre_fusion": 13,
        "cold_ms": 1.0,
        "counter_delta": {
            "aot_autograd": {"total": 1, "ok": 1},
            "inductor": {"extern_calls": 3},
        },
    }
    return {
        "candidate_vs_reference": delta,
        "control_vs_reference": None,
        "candidate_vs_control": None,
        "cotangent": {
            "candidate_delta": 0.0,
            "control_delta": None,
            "candidate_version": [0, 0],
            "control_version": None,
        },
        "peak_allocated_bytes": {"candidate": 1},
        "hashes": {"candidate_gx": "a" * 64},
        "compiled": {
            "candidate_vs_reference": delta,
            "control_vs_reference": None,
            "candidate_eager_vs_compiled": delta,
            "control_eager_vs_compiled": None,
            "candidate_observation": {
                "cotangent_delta": 0.0,
                "cotangent_version": [0, 0],
                "peak_allocated_bytes": 1,
            },
            "control_observation": None,
            "compile": {"candidate": occurrence},
        },
        "environment": {
            "python": "3.14.6",
            "platform": "test",
            "torch": "2.12.1+cu130",
            "torch_git": "7269437d655783a26cba32aa88195b741ff496aa",
            "triton": "3.7.1",
            "cuda_runtime": "test",
            "device": "test-device",
            "capability": [1, 0],
            "total_memory_bytes": 0,
            "multiprocessors": 0,
            "driver": None,
        },
        "shape": [2, 8192],
        "seed": 17,
    }


def _generic_art_result():
    return {
        "valid": True,
        "outcome": "SAFE_IN_FRAGMENT",
        "reasons": ["FRESH_DESTINATION"],
        "borrowed_writes": 0,
        "contextual_collapse": False,
    }


def _art_generic_runtime(runtime):
    return {
        "environment": runtime["environment"],
        "candidate": {
            "compiled": {"occurrence": runtime["compiled"]["compile"]["candidate"]}
        },
        "control": None,
    }


def test_art_runtime_evidence_uses_generic_nullable_control_schema(monkeypatch):
    lane = {"local_sum": object(), "gradient": object(), "peak_allocated_bytes": 3,
            "cotangent_delta": 0.0, "cotangent_version": [0, 0]}
    monkeypatch.setattr(art_target, "_reference_art_lane", lambda *_args: lane)
    monkeypatch.setattr(art_target, "_run_art_lane", lambda *_args: lane)
    monkeypatch.setattr(art_target, "_compile_art_lane", lambda *_args: (lane, {
        "generated_kernel_count": 1,
        "ir_nodes_pre_fusion": 13,
        "cold_ms": 1.0,
        "counter_delta": {"aot_autograd": {"total": 1, "ok": 1}, "inductor": {"extern_calls": 3}},
    }))
    monkeypatch.setattr(art_target, "_max_abs", lambda *_args: 0.0)
    monkeypatch.setattr(art_target, "_tensor_hash", lambda *_args: "a" * 64)
    environment = _art_runtime_fixture()["environment"]
    monkeypatch.setattr(art_target, "_gpu_environment", lambda _target_name: environment)

    runtime, target_runtime = art_target._runtime_evidence_pair(
        object(), object(), object()
    )

    replay_runtime_evidence(runtime)
    assert set(runtime) == {"environment", "candidate", "control"}
    assert runtime["control"] is None
    assert set(runtime["candidate"]) == {"compiled"}
    assert set(runtime["candidate"]["compiled"]) == {"occurrence"}
    assert '"compile":' not in json.dumps(runtime)
    art_target._validate_art_runtime_witness(target_runtime)
    target_runtime["hashes"]["unexpected"] = "x"
    with pytest.raises(TargetError, match="ART runtime hashes fields changed"):
        art_target._validate_art_runtime_witness(target_runtime)


@pytest.mark.parametrize(
    "path",
    (
        ("candidate_vs_reference", "output_max_abs"),
        ("candidate_vs_reference", "gx_max_abs"),
        ("compiled", "candidate_vs_reference", "output_max_abs"),
        ("compiled", "candidate_vs_reference", "gx_max_abs"),
        ("compiled", "candidate_eager_vs_compiled", "output_max_abs"),
        ("compiled", "candidate_eager_vs_compiled", "gx_max_abs"),
    ),
)
@pytest.mark.parametrize("replacement", (float("nan"), float("inf"), -1.0))
def test_art_runtime_evidence_rejects_nonfinite_or_negative_deltas(path, replacement):
    comparison = copy.deepcopy(_art_runtime_fixture())
    owner = comparison
    for key in path[:-2]:
        owner = owner[key]
    parent_key, field = path[-2:]
    owner[parent_key] = {**owner[parent_key], field: replacement}

    with pytest.raises(TargetError):
        art_target._validate_art_runtime_evidence(comparison)


@pytest.mark.parametrize("value", (0.0, 2e-4))
def test_art_runtime_evidence_accepts_zero_and_tolerance_deltas(value):
    comparison = copy.deepcopy(_art_runtime_fixture())
    comparison["candidate_vs_reference"] = {
        "output_max_abs": value,
        "gx_max_abs": value,
        "gw_max_abs": 0.0,
    }
    comparison["compiled"]["candidate_vs_reference"] = {
        "output_max_abs": value,
        "gx_max_abs": value,
        "gw_max_abs": 0.0,
    }
    comparison["compiled"]["candidate_eager_vs_compiled"] = {
        "output_max_abs": value,
        "gx_max_abs": value,
        "gw_max_abs": 0.0,
    }

    art_target._validate_art_runtime_evidence(comparison)


def test_art_sidecars_use_v2_capture_and_canonical_null_control(monkeypatch, tmp_path):
    root, _expected = _exact_tree_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(art_target._certificate, "sha256_bytes", source_files.sha256_bytes)
    monkeypatch.setattr(art_target._certificate, "replay_certificate", lambda _path: {"valid": True})
    packet = tmp_path / "packet"
    candidate = _art_candidate()
    analysis = _ownership.analyze(candidate)
    target_runtime = _art_runtime_fixture()
    runtime = _art_generic_runtime(target_runtime)
    benchmark = {"warmups": 1, "samples": 3, "lanes": {}}
    art_target._stage_art_sources(root, packet)
    target = art_target._target_identity(
        root,
        candidate,
        _art_observations(candidate),
        runtime_witness=target_runtime,
    )
    art_target._write_art_sidecars(packet, candidate, analysis, runtime, benchmark, target)

    candidate_payload = json.loads((packet / "candidate.capture.json").read_text())
    certificate = json.loads((packet / "certificate.json").read_text())
    body = {key: value for key, value in certificate.items() if key != "digest"}

    assert candidate_payload["schema"] == CAPTURE_SCHEMA
    assert set(candidate_payload["diagnostics"]) == {
        "callbacks",
        "aot_autograd",
        "inputs",
        "grad_output",
    }
    assert json.loads((packet / "control.capture.json").read_text()) is None
    assert certificate["schema"] == CERTIFICATE_SCHEMA
    assert certificate["control"] is None
    assert certificate["controls"] == {}
    assert json.loads((packet / "controls.json").read_text()) == {}
    assert certificate["validation"] == {
        "oracle_cases": 0,
        "oracle_passed": 0,
        "mutation_cases": 0,
        "mutations_killed": 0,
        "ablations": 0,
        "ablations_observed": 0,
    }
    assert certificate["digest"] == sha256_json(body)
    assert not (packet / "analysis.json").exists()
    assert not (packet / "target.json").exists()
    compile((packet / "reproducer.py").read_text(), "reproducer.py", "exec")
    assert {
        path.relative_to(packet).as_posix()
        for path in packet.rglob("*")
        if path.is_file()
    } == {
        "certificate.json",
        "candidate.capture.json",
        "control.capture.json",
        "runtime.json",
        "benchmark.json",
        "controls.json",
        "reproducer.py",
        "external/openpipe-art/tree/LICENSE",
        "external/openpipe-art/tree/THIRD-PARTY-NOTICES",
        "external/openpipe-art/tree/licenses/GPL-3.0.txt",
        "external/openpipe-art/tree/licenses/LGPL-3.0.txt",
        "external/openpipe-art/tree/src/art/trainer_rank/topk.py",
    }


def test_art_adapter_binds_target_closure_to_the_generic_loaded_certificate(
    monkeypatch, tmp_path
):
    certificate_path = tmp_path / "packet" / "certificate.json"
    loaded = {"target": {"repository": art_target.ART_REPOSITORY}}
    generic = {
        "valid": True,
        "certificate_digest": "a" * 64,
        "outcome": "SAFE_IN_FRAGMENT",
        "reasons": ["FRESH_DESTINATION"],
        "borrowed_writes": 0,
        "contextual_collapse": False,
    }
    observed = []

    def closure(certificate, path, generic_result):
        observed.append((certificate, path, generic_result))
        return {"adapter": art_target.ART_TARGET_ID}

    def generic_replay(path, *, certificate_validator):
        updates = certificate_validator(loaded, path, generic)
        return {**generic, **updates}

    monkeypatch.setattr(
        cli_replay,
        "_replay_certificate_with_validator",
        generic_replay,
    )
    monkeypatch.setattr(art_target, "_replay_art_loaded", closure)

    assert replay_certificate(certificate_path)["adapter"] == (
        art_target.ART_TARGET_ID
    )
    assert observed == [(loaded, certificate_path, generic)]


def test_art_runtime_binding_rejects_a_noncanonical_config():
    candidate = _art_candidate()
    candidate.kernels[0].runtime["config"]["num_stages"] = 2

    with pytest.raises(InvalidArtifact, match="exact ART runtime profile"):
        art_target._validate_art_runtime_bindings(candidate, _art_observations(candidate))


def test_art_runtime_binding_rejects_a_malformed_observation_cache_key():
    candidate = _art_candidate()
    observations = _art_observations(candidate)
    observations[0]["jit_cache_key"] = "not-a-cache-key"

    with pytest.raises(UnsupportedCapture, match="cache keys"):
        art_target._validate_art_runtime_bindings(candidate, observations)


def test_art_runtime_binding_rejects_wrong_hop_multiplicity():
    candidate = _art_candidate()
    hop = next(node for node in candidate.forward.nodes if node.target == "triton.functional")
    forward = replace(candidate.forward, nodes=(*candidate.forward.nodes, replace(hop, id="n99")))

    with pytest.raises(UnsupportedCapture, match="HOP multiplicity"):
        art_target._validate_art_runtime_bindings(
            replace(candidate, forward=forward), _art_observations(candidate)
        )


def test_art_replay_rejects_wrong_target_specialization(monkeypatch, tmp_path):
    root, _expected = _exact_tree_fixture(monkeypatch, tmp_path)
    packet = tmp_path / "packet"
    packet.mkdir()
    candidate = _art_candidate()
    observations = _art_observations(candidate)
    art_target._stage_art_sources(root, packet)
    runtime_witness = _art_runtime_fixture()
    target = art_target._target_identity(
        root,
        candidate,
        observations,
        runtime_witness=runtime_witness,
    )
    target["specialization"]["expression"] = "resealed"
    raw = {
        "target": target,
        "candidate": candidate.to_dict(),
        "runtime": _art_generic_runtime(runtime_witness),
    }

    with pytest.raises(TargetError, match="target identity"):
        art_target._replay_art_loaded(
            raw, packet / "certificate.json", _generic_art_result()
        )


def test_art_public_replay_rejects_a_resealed_capture_specialization(tmp_path):
    source = Path(__file__).parents[1] / "evidence" / "results" / "bundle" / "art" / "openpipe-art-local-stats"
    packet = tmp_path / "packet"
    shutil.copytree(source, packet)
    certificate_path = packet / "certificate.json"
    certificate = json.loads(certificate_path.read_text())
    candidate = copy.deepcopy(certificate["candidate"])
    candidate["diagnostics"]["inputs"][0]["dtype"] = "float16"
    certificate["candidate"] = candidate
    certificate["analysis"]["candidate_digest"] = sha256_json(candidate)
    write_canonical_json(packet / "candidate.capture.json", candidate)
    body = {key: value for key, value in certificate.items() if key != "digest"}
    certificate["digest"] = sha256_json(body)
    write_canonical_json(certificate_path, certificate)

    with pytest.raises(ReplayError, match="ART packet could not be replayed"):
        replay_certificate(certificate_path)


def test_art_replay_rejects_a_runtime_projection_mismatch(monkeypatch, tmp_path):
    root, _expected = _exact_tree_fixture(monkeypatch, tmp_path)
    packet = tmp_path / "packet"
    packet.mkdir()
    candidate = _art_candidate()
    observations = _art_observations(candidate)
    art_target._stage_art_sources(root, packet)
    runtime_witness = _art_runtime_fixture()
    target = art_target._target_identity(
        root,
        candidate,
        observations,
        runtime_witness=runtime_witness,
    )
    runtime = copy.deepcopy(_art_generic_runtime(runtime_witness))
    runtime["candidate"]["compiled"]["occurrence"]["generated_kernel_count"] = 2
    raw = {
        "target": target,
        "candidate": candidate.to_dict(),
        "runtime": runtime,
    }

    with pytest.raises(TargetError, match="generic runtime lane"):
        art_target._replay_art_loaded(
            raw, packet / "certificate.json", _generic_art_result()
        )


def test_art_reproducer_rejects_cpu_runtime_before_fetch_or_allocation(monkeypatch, tmp_path):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(
        art_target,
        "_require_module",
        lambda name, _display_name, _target_name: fake_torch if name == "torch" else object(),
    )

    with pytest.raises(TargetError, match="reproducer requires CUDA"):
        art_target.run_art_reproducer(tmp_path)


def test_art_reproducer_requires_an_explicit_root_before_any_fetch(monkeypatch):
    monkeypatch.setattr(
        art_target,
        "fetch_art",
        lambda *_args: (_ for _ in ()).throw(AssertionError("network reached")),
    )

    with pytest.raises(TargetError, match="explicit target_root"):
        art_target.run_art_reproducer()


@pytest.mark.parametrize(
    ("warmups", "samples"),
    [(True, 3), (0, 3), (11, 3), (1, True), (1, 2), (1, 51)],
)
def test_art_audit_rejects_unbounded_benchmark_before_runtime(
    monkeypatch, tmp_path, warmups, samples
):
    monkeypatch.setattr(
        art_target,
        "_require_module",
        lambda *_args: (_ for _ in ()).throw(AssertionError("runtime import reached")),
    )

    with pytest.raises(TargetError, match="ART benchmark (warmups|samples)"):
        art_target.audit_art(tmp_path / "audit", warmups=warmups, samples=samples)
