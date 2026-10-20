from __future__ import annotations

from pathlib import Path

import pytest

from daggertrace import cli, cli_audit, cli_output, cli_replay
from daggertrace.cli import main
from daggertrace.errors import DaggerTraceError
from daggertrace.replay import ReplayError
from daggertrace.triton_protocols import UNKNOWN_MEMORY_MODEL_DETAIL

class _EncodingStream:
    def __init__(self, encoding):
        self.encoding = encoding
        self.value = ""

    def write(self, value):
        value.encode(self.encoding)
        self.value += value
        return len(value)


@pytest.mark.parametrize(
    ("encoding", "expected"),
    [
        ("utf-8", "DaggerTrace — café |Δgx|\n"),
        ("ascii", "DaggerTrace \\u2014 caf\\xe9 |\\u0394gx|\n"),
    ],
)
def test_write_text_preserves_utf8_and_escapes_ascii(encoding, expected):
    stream = _EncodingStream(encoding)

    cli_output._write_text("DaggerTrace — café |Δgx|", stream)

    assert stream.value == expected


def test_replay_error_text_is_ascii_safe(monkeypatch):
    stream = _EncodingStream("ascii")
    monkeypatch.setattr(
        cli_replay,
        "replay_certificate",
        lambda _path: (_ for _ in ()).throw(ReplayError("café")),
    )
    monkeypatch.setattr(cli.sys, "stderr", stream)

    assert main(["replay", "unused.json"]) == 1
    assert stream.value == "daggertrace: replay rejected: caf\\xe9\n"


def test_gpu_commands_require_explicit_acknowledgement(capsys):
    assert main(["audit", "liger-rmsnorm"]) == 2
    assert "requires explicit --gpu" in capsys.readouterr().err


def test_source_only_target_is_inspection_only(monkeypatch, tmp_path):
    from daggertrace import memory_model_target

    result = memory_model_target.audit_triton_layer_norm(
        Path(__file__).parent / "fixtures" / "external" / "triton"
    )
    observed = []
    monkeypatch.setattr(
        memory_model_target,
        "audit_triton_layer_norm",
        lambda root: observed.append(root) or result,
    )
    emitted = []
    monkeypatch.setattr(cli, "_emit", lambda value, *_args: emitted.append(value))

    assert main(
        ["inspect", "triton-layernorm", "--target-root", str(tmp_path), "--json"]
    ) == 0
    assert observed == [tmp_path.resolve()]
    assert emitted == [result]


@pytest.mark.parametrize(
    "case", ("malformed", "empty", "wrong", "opposite", "target", "evidence")
)
def test_source_inspection_rejects_invalid_fixed_results(monkeypatch, tmp_path, capsys, case):
    from daggertrace import memory_model_target
    from daggertrace.memory_model_target import _classify_source

    valid = memory_model_target.audit_triton_layer_norm(
        Path(__file__).parent / "fixtures" / "external" / "triton"
    )
    if case == "malformed":
        result = []
    elif case == "empty":
        result = {}
    elif case == "wrong":
        result = {
            **valid,
            "reason": "SOURCE_PROFILE",
            "status": "UNKNOWN:SOURCE_PROFILE",
            "detail": "backward contains calls outside the supported source subset",
        }
    elif case == "opposite":
        opposite = _classify_source(
            """\
def _layer_norm_bwd_dx_fused(pointer):
    value = tl.load(pointer)
    tl.store(pointer, value)

def _layer_norm_bwd_dwdb(pointer):
    pass

class LayerNorm:
    @staticmethod
    def backward(ctx, dy):
        _layer_norm_bwd_dx_fused[0](dy)
        _layer_norm_bwd_dwdb[0](dy)
"""
        )
        result = {
            **opposite,
            "schema": valid["schema"],
            "target": valid["target"],
            "provenance": valid["provenance"],
        }
    elif case == "target":
        result = {**valid, "target": {}}
    else:
        result = {**valid, "evidence": {}}
    emitted = []
    monkeypatch.setattr(memory_model_target, "audit_triton_layer_norm", lambda _root: result)
    monkeypatch.setattr(cli, "_emit", lambda *args: emitted.append(args))

    assert main(
        ["inspect", "triton-layernorm", "--target-root", str(tmp_path)]
    ) == 2
    assert "source classification" in capsys.readouterr().err
    assert not emitted


def test_audit_all_excludes_the_source_only_target():
    assert cli_audit._selected_targets("all") == (
        "liger-rmsnorm",
        "openpipe-art-local-stats",
    )


@pytest.mark.parametrize("command", ("audit", "reproduce"))
def test_source_only_target_is_inspect_only(command, capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command, "triton-layernorm"])
    assert "invalid choice" in capsys.readouterr().err


def test_source_inspection_requires_a_pinned_checkout(capsys):
    assert main(["inspect", "triton-layernorm"]) == 2
    assert capsys.readouterr().err == (
        "daggertrace: triton-layernorm requires --target-root for its pinned source checkout\n"
    )


def test_source_inspection_text_is_not_an_aot_audit():
    rendered = cli_output._source_classification_text(
        "triton-layernorm",
        {
            "outcome": "UNKNOWN",
            "reason": "MEMORY_MODEL",
            "status": "UNKNOWN:MEMORY_MODEL",
            "detail": UNKNOWN_MEMORY_MODEL_DETAIL,
            "target": {
                "revision": "a" * 40,
                "files": {"source": {"path": "tutorial.py", "sha256": "b" * 64}},
            },
            "evidence": {
                "reachable_kernels": [
                    "_layer_norm_bwd_dx_fused",
                    "_layer_norm_bwd_dwdb",
                ]
            },
        },
    )

    assert rendered.startswith("DaggerTrace — source-only classification")
    assert "CLASSIFICATION ONLY  no AOT capture or GPU execution" in rendered
    assert f"SOURCE       tutorial.py  sha256:{'b' * 64}" in rendered
    assert "AOT/Triton ownership" not in rendered


@pytest.mark.parametrize(
    ("files", "expected_sources"),
    (
        (
            [
                {
                    "packet_path": "external/art/tree/src/art/trainer_rank/topk.py",
                    "target_path": "src/art/trainer_rank/topk.py",
                    "size": 1,
                    "sha256": "a" * 64,
                    "role": "source",
                },
                {
                    "packet_path": "external/art/tree/NOTICE",
                    "target_path": "NOTICE",
                    "size": 1,
                    "sha256": "b" * 64,
                    "role": "support",
                },
                {
                    "packet_path": "external/art/tree/LICENSE",
                    "target_path": "LICENSE",
                    "size": 1,
                    "sha256": "c" * 64,
                    "role": "license",
                },
            ],
            (("src/art/trainer_rank/topk.py", "a" * 64),),
        ),
        (
            [
                {
                    "packet_path": "external/first/tree/kernel.py",
                    "target_path": "kernel.py",
                    "size": 1,
                    "sha256": "d" * 64,
                    "role": "source",
                },
                {
                    "packet_path": "external/second/tree/helper.py",
                    "target_path": "helper.py",
                    "size": 1,
                    "sha256": "e" * 64,
                    "role": "source",
                },
            ],
            (("kernel.py", "d" * 64), ("helper.py", "e" * 64)),
        ),
        (
            [
                {
                    "packet_path": "external/art/tree/NOTICE",
                    "target_path": "NOTICE",
                    "size": 1,
                    "sha256": "f" * 64,
                    "role": "support",
                },
                {
                    "packet_path": "external/art/tree/LICENSE",
                    "target_path": "LICENSE",
                    "size": 1,
                    "sha256": "0" * 64,
                    "role": "license",
                },
            ],
            (),
        ),
    ),
)
def test_audit_text_renders_all_certificate_source_entries(files, expected_sources):
    rendered = cli_output._audit_text(
        {
            "target_id": "openpipe-art-local-stats",
            "target": {"revision": "a" * 40, "files": files},
            "analysis": {"outcome": "SAFE_IN_FRAGMENT", "reasons": []},
            "artifacts": {},
            "status": "SAFE_IN_FRAGMENT",
        }
    )

    source_lines = [line for line in rendered.splitlines() if line.startswith("SOURCE       ")]
    assert source_lines == [
        f"SOURCE       {path}  sha256:{digest}" for path, digest in expected_sources
    ]


def test_validation_text_distinguishes_recorded_mutations_from_executed_checks():
    rendered = cli_output._validation_text(
        {
            "oracle_passed": 48,
            "oracle_cases": 48,
            "mutation_cases": 34,
            "ablations_observed": 7,
            "ablations": 7,
        }
    )

    assert "recorded packet mutation tests 34 (not rerun)" in rendered
    assert "certificate mutation denominator" not in rendered


def test_replay_mutation_converts_target_rejection_to_replay_error(monkeypatch, tmp_path):
    failure = DaggerTraceError("target closure rejected")

    def reject(_path):
        raise failure

    monkeypatch.setattr(cli_replay, "_replay_checked", reject)

    with pytest.raises(ReplayError, match="target closure rejected") as raised:
        cli_replay._replay_mutation(tmp_path / "certificate.json")
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize("arguments", [
    ["seal", "raw.json"],
    ["ingest", "raw.json"],
    ["audit", "daggertrace-calibrated-softmax", "--gpu"],
    ["replay", "packet", "--target-neutral"],
])
def test_retired_workflows_are_rejected(arguments):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(arguments)
    assert error.value.code == 2
