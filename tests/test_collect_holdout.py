from __future__ import annotations

import subprocess
import sys

import pytest

import evidence.collect_holdout as holdout


_TARGET_ADAPTER_MODULES = (
    "daggertrace.targets",
    "daggertrace.art_capture",
    "daggertrace.art_target",
    "daggertrace.liger_capture",
    "daggertrace.liger_packet",
    "daggertrace.liger_target",
    "daggertrace.memory_model_target",
)


@pytest.fixture(autouse=True)
def _target_free_test_environment(monkeypatch):
    """Keep normal collection tests independent of pytest's eager module collection."""

    package = sys.modules["daggertrace"]
    for module in _TARGET_ADAPTER_MODULES:
        monkeypatch.delitem(sys.modules, module, raising=False)
        monkeypatch.delattr(package, module.rpartition(".")[2], raising=False)


def _patch_cpu_collection(monkeypatch, *, missing=None, nonfinite=False):
    monkeypatch.setattr(holdout, "_source_state", lambda: {"stable": True})
    monkeypatch.setattr(holdout, "_load_module", lambda *_args: object())

    def baseline(_module, size):
        return {
            "size": size,
            "output_max_abs": 0.0,
            "gradient_max_abs": 0.0,
        }

    def composed(_module, size):
        return {
            "size": size,
            "numeric": {
                "safe": {
                    "output_max_abs": 0.0,
                    "gradient_vs_reference_max_abs": 0.0,
                    "cotangent_delta_max_abs": 0.0,
                },
                "unsafe": {
                    "output_max_abs": 0.0,
                    "gradient_vs_reference_max_abs": 1.0,
                    "cotangent_delta_max_abs": 1.0,
                },
                "unsafe_vs_safe_gradient_max_abs": 1.0,
            },
        }

    def affine(_module, size):
        return {
            "size": size,
            "numeric": {
                "output_max_abs": 0.0,
                "gradient_x_max_abs": 0.0,
                "gradient_weight_max_abs": 0.0,
            },
        }

    def reduction(_module, size):
        return {
            "size": size,
            "numeric": {"output_max_abs": 0.0, "gradient_max_abs": 0.0},
        }

    runners = {
        "baseline": baseline,
        "composed": composed,
        "affine": affine,
        "reduction": reduction,
    }
    for name, runner in runners.items():
        if name == nonfinite:
            original = runner

            def runner(_module, size, original=original, family_name=name):
                result = original(_module, size)
                if family_name == "baseline":
                    result["output_max_abs"] = float("nan")
                else:
                    result["numeric"]["safe"]["output_max_abs"] = float("nan")
                return result

        if name == missing:
            def runner(_module, _size):
                return None
        function_name = {
            "baseline": "_run_shape",
            "composed": "_run_composed_shape",
            "affine": "_run_affine_shape",
            "reduction": "_run_reduction_shape",
        }[name]
        monkeypatch.setattr(holdout, function_name, runner)


def _temporary_holdout_sources(monkeypatch, tmp_path):
    sources = (
        tmp_path / "holdout_kernel.py",
        tmp_path / "affine_holdout.py",
        tmp_path / "reduction_holdout.py",
    )
    for index, source in enumerate(sources):
        source.write_text(f"VALUE = {index}\n", encoding="utf-8")
    monkeypatch.setattr(holdout, "REPO", tmp_path)
    monkeypatch.setattr(holdout, "HOLDOUT", sources[0])
    monkeypatch.setattr(holdout, "AFFINE_HOLDOUT", sources[1])
    monkeypatch.setattr(holdout, "REDUCTION_HOLDOUT", sources[2])
    return sources


def _valid_conclusion():
    return {
        "all_numeric_checks_passed": True,
        "baseline_safe_cases_passed": True,
        "composed_control_safe": True,
        "composed_candidate_violation_detected": True,
        "composed_contribution_collapse_detected": True,
        "multi_input_saved_state_cases_passed": True,
        "compiled_affine_case_passed": True,
        "reduction_cases_passed": True,
        "compiled_reduction_cases_passed": True,
        "generalization_scope": "the existing closed AOT/Triton fragment, not arbitrary Triton",
    }


def test_holdout_source_state_records_exact_head(monkeypatch, tmp_path):
    source_root = tmp_path / "src" / "daggertrace"
    source_root.mkdir(parents=True)
    (source_root / "capture.py").write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(holdout, "REPO", tmp_path)

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments], cwd=tmp_path, check=True, capture_output=True, text=True,
        ).stdout.strip()

    git("init", "-q")
    git("add", ".")
    commit_options = (
        "-c", "user.name=DaggerTrace tests", "-c", "user.email=tests@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm",
    )
    git(*commit_options, "record source")
    source_revision = git("rev-parse", "HEAD")
    (tmp_path / "README.md").write_text("documentation only\n", encoding="utf-8")
    git("add", "README.md")
    git(*commit_options, "record documentation")
    head = git("rev-parse", "HEAD")
    assert head != source_revision

    state = holdout._source_state()

    assert state["commit"] == head
    assert state["source_tree_dirty"] is False
    source = source_root / "capture.py"
    source.write_text("value = 2\n", encoding="utf-8")
    state = holdout._source_state()

    assert state["commit"] == head
    assert state["source_tree_dirty"] is True


def test_holdout_source_state_uses_source_revision_in_archive(monkeypatch, tmp_path):
    source_root = tmp_path / "src" / "daggertrace"
    source_root.mkdir(parents=True)
    (source_root / "capture.py").write_text("value = 1\n", encoding="utf-8")
    revision = "a" * 40
    (tmp_path / "SOURCE_REVISION").write_text(revision + "\n", encoding="utf-8")
    monkeypatch.setattr(holdout, "REPO", tmp_path)

    state = holdout._source_state()

    assert state["commit"] == revision
    assert state["source_tree_dirty"] is None


@pytest.mark.parametrize(
    ("marker", "message"),
    (
        (None, "Git metadata"),
        ("not-a-revision\n", "lowercase 40-hex"),
        ("A" * 40 + "\n", "lowercase 40-hex"),
        ("a" * 39 + "\n", "lowercase 40-hex"),
        ("\n", "lowercase 40-hex"),
    ),
)
def test_holdout_source_state_rejects_missing_or_malformed_archive_revision(
    monkeypatch, tmp_path, marker, message
):
    source_root = tmp_path / "src" / "daggertrace"
    source_root.mkdir(parents=True)
    (source_root / "capture.py").write_text("value = 1\n", encoding="utf-8")
    if marker is not None:
        (tmp_path / "SOURCE_REVISION").write_text(marker, encoding="utf-8")
    monkeypatch.setattr(holdout, "REPO", tmp_path)

    with pytest.raises(RuntimeError, match=message):
        holdout._source_state()


def test_holdout_rejects_empty_and_out_of_bounds_sizes_before_runtime(monkeypatch):
    monkeypatch.setattr(
        holdout,
        "_source_state",
        lambda: pytest.fail("size validation must precede runtime setup"),
    )

    with pytest.raises(ValueError, match="at least one"):
        holdout.collect(())
    with pytest.raises(ValueError, match="8192"):
        holdout.collect((8193,))
    with pytest.raises(ValueError, match="duplicates"):
        holdout.collect((1, 1))


@pytest.mark.parametrize("missing", ("baseline", "composed", "affine", "reduction"))
def test_holdout_requires_each_family_to_cover_each_requested_size(monkeypatch, missing):
    _patch_cpu_collection(monkeypatch, missing=missing)

    with pytest.raises(RuntimeError, match="family coverage is incomplete"):
        holdout.collect((1, 2))


def test_holdout_rejects_nonfinite_numeric_evidence(monkeypatch):
    _patch_cpu_collection(monkeypatch, nonfinite="composed")

    with pytest.raises(RuntimeError, match="non-finite"):
        holdout.collect((1,))


def test_holdout_direct_collection_has_success_and_failure_statuses(monkeypatch):
    _patch_cpu_collection(monkeypatch)

    result = holdout.collect((1,))
    assert result["proof_valid"] is True

    monkeypatch.setattr(holdout, "_run_reduction_shape", lambda _module, _size: None)
    with pytest.raises(RuntimeError, match="family coverage is incomplete"):
        holdout.collect((1,))


def test_holdout_rejects_source_changed_by_loader_before_execution(monkeypatch, tmp_path):
    real_load_module = holdout._load_module
    _patch_cpu_collection(monkeypatch)
    holdout_source, _, _ = _temporary_holdout_sources(monkeypatch, tmp_path)

    def load_module(path, name):
        module = real_load_module(path, name)
        if path == holdout_source:
            path.write_text("VALUE = 'changed'\n", encoding="utf-8")
        return module

    monkeypatch.setattr(holdout, "_load_module", load_module)
    monkeypatch.setattr(
        holdout,
        "_run_shape",
        lambda *_args: pytest.fail("source drift must be rejected before execution"),
    )

    with pytest.raises(RuntimeError, match=holdout_source.name):
        holdout.collect((1,))


def test_holdout_rejects_source_changed_during_final_execution(monkeypatch, tmp_path):
    _patch_cpu_collection(monkeypatch)
    holdout_source, _, _ = _temporary_holdout_sources(monkeypatch, tmp_path)

    def run_reduction_shape(_module, _size):
        holdout_source.write_text("VALUE = 'changed'\n", encoding="utf-8")
        return {
            "size": 1,
            "numeric": {"output_max_abs": 0.0, "gradient_max_abs": 0.0},
        }

    monkeypatch.setattr(holdout, "_run_reduction_shape", run_reduction_shape)

    with pytest.raises(RuntimeError, match=holdout_source.name):
        holdout.collect((1,))


def test_holdout_emits_initial_hashes_when_sources_are_unchanged(monkeypatch, tmp_path):
    _patch_cpu_collection(monkeypatch)
    holdout_source, affine_source, reduction_source = _temporary_holdout_sources(monkeypatch, tmp_path)
    initial_hashes = {
        source.relative_to(tmp_path).as_posix(): holdout.sha256_file(source)
        for source in (holdout_source, affine_source, reduction_source)
    }

    result = holdout.collect((1,))

    assert result["holdout"]["sha256"] == initial_hashes[holdout_source.name]
    assert {entry["path"]: entry["sha256"] for entry in result["holdout"]["modules"]} == (
        initial_hashes
    )


@pytest.mark.parametrize("adapter_module", _TARGET_ADAPTER_MODULES)
def test_holdout_rejects_a_preexisting_adapter_without_mutating_it(monkeypatch, adapter_module):
    monkeypatch.setattr(
        holdout,
        "_source_state",
        lambda: pytest.fail("target adapter preflight must precede runtime setup"),
    )
    preexisting = object()
    package = sys.modules["daggertrace"]
    attribute = adapter_module.rpartition(".")[2]
    monkeypatch.setitem(sys.modules, adapter_module, preexisting)
    monkeypatch.setattr(package, attribute, preexisting, raising=False)

    with pytest.raises(RuntimeError, match="target adapters"):
        holdout.collect((1,))

    assert sys.modules[adapter_module] is preexisting
    assert getattr(package, attribute) is preexisting


@pytest.mark.parametrize("adapter_module", _TARGET_ADAPTER_MODULES)
def test_holdout_rejects_an_adapter_imported_during_module_load(monkeypatch, adapter_module):
    _patch_cpu_collection(monkeypatch)

    def load_module(*_args):
        monkeypatch.setitem(sys.modules, adapter_module, object())
        return object()

    monkeypatch.setattr(holdout, "_load_module", load_module)

    with pytest.raises(RuntimeError, match="target adapters"):
        holdout.collect((1,))


@pytest.mark.parametrize("adapter_module", _TARGET_ADAPTER_MODULES)
def test_holdout_rejects_an_adapter_imported_during_final_shape_execution(
    monkeypatch, adapter_module
):
    _patch_cpu_collection(monkeypatch)

    def run_reduction_shape(_module, _size):
        monkeypatch.setitem(sys.modules, adapter_module, object())
        return {
            "size": 1,
            "numeric": {"output_max_abs": 0.0, "gradient_max_abs": 0.0},
        }

    monkeypatch.setattr(holdout, "_run_reduction_shape", run_reduction_shape)

    with pytest.raises(RuntimeError, match="target adapters"):
        holdout.collect((1,))


@pytest.mark.parametrize(
    "mutation",
    ("missing", "extra", "false", "nonbool", "scope"),
)
def test_holdout_proof_gate_rejects_incomplete_or_mistyped_conclusion(
    monkeypatch, tmp_path, mutation
):
    conclusion = _valid_conclusion()
    if mutation == "missing":
        conclusion.pop("reduction_cases_passed")
    elif mutation == "extra":
        conclusion["unexpected"] = True
    elif mutation == "false":
        conclusion["reduction_cases_passed"] = False
    elif mutation == "nonbool":
        conclusion["reduction_cases_passed"] = "true"
    else:
        conclusion["generalization_scope"] = None

    assert holdout._proof_valid(conclusion) is False
    monkeypatch.setattr(holdout, "collect", lambda _sizes: {
        "cases": [], "composed_cases": [], "affine_cases": [], "reduction_cases": [],
        "conclusion": conclusion, "proof_valid": True,
    })
    monkeypatch.setattr(sys, "argv", [
        "collect_holdout", "--output", str(tmp_path / "holdout.json"), "--sizes", "1",
    ])

    assert holdout.main() == 1


def test_holdout_proof_gate_accepts_the_complete_conclusion():
    assert holdout._proof_valid(_valid_conclusion()) is True


@pytest.mark.parametrize("proof_valid,expected", ((False, 1), (None, 1), (True, 0)))
def test_holdout_main_propagates_proof_status(monkeypatch, tmp_path, proof_valid, expected):
    monkeypatch.setattr(
        holdout,
        "collect",
        lambda _sizes: {
            "cases": [],
            "composed_cases": [],
            "affine_cases": [],
            "reduction_cases": [],
            "conclusion": _valid_conclusion(),
            "proof_valid": proof_valid,
        },
    )
    monkeypatch.setattr(holdout, "write_canonical_json", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_holdout",
            "--output",
            str(tmp_path / "holdout.json"),
            "--sizes",
            "1",
        ],
    )

    assert holdout.main() == expected
