from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

import daggertrace.validation as validation_module
from daggertrace.canonical import canonical_json_bytes, sha256_bytes, sha256_json
from daggertrace.effects import profile_triton_source
from daggertrace.errors import InvalidArtifact
from daggertrace.model import Capture, Outcome, Reason
from daggertrace.ownership import analyze
from daggertrace.replay import ReplayError
from daggertrace.validation import run_semantic_validation
from daggertrace.validation_mutations import (
    _certificate_mutations,
    _corrupt_hop_binding,
    _corruption_mutations,
    _closure_mutations,
    _evidence_integrity_mutations,
    validate_certificate_mutations,
)
from daggertrace.validation_v2 import (
    ABLATION_COUNT,
    CLOSURE_MUTATION_COUNT,
    CORRUPTION_MUTATION_COUNT,
    EVIDENCE_INTEGRITY_MUTATION_COUNT,
    V2_MUTATION_COUNT,
    V2_ORACLE_CASES,
    _ablation_cases,
    _expected_micrograph,
    _matches_expected,
    _semantic_cases,
    _run_ablations,
)
from daggertrace.validation_v2_fixtures import (
    _alpha_rename_capture,
    _fresh_backward_allocation,
    _micrograph,
    _relabel_capture,
    _unsupported_kernel_capture,
)


def _semantic_projection(result) -> tuple[Outcome, tuple[str, ...], int, bool]:
    return (
        result.outcome,
        tuple(reason.value for reason in result.reasons),
        len(result.borrowed_writes),
        result.collapse is not None,
    )


def test_exhaustive_micrograph_oracle_and_ablations():
    result = run_semantic_validation()
    assert result == {
        "oracle_cases": 48,
        "oracle_passed": 48,
        "mutation_cases": 34,
        "ablations": 7,
        "ablations_observed": 7,
    }
    assert result["oracle_cases"] == result["oracle_passed"]
    assert result["oracle_cases"] == V2_ORACLE_CASES
    assert result["mutation_cases"] == V2_MUTATION_COUNT
    assert len(_certificate_mutations(_certificate_fixture())) == V2_MUTATION_COUNT
    assert result["ablations"] == result["ablations_observed"] == len(_ablation_cases())
    oracle_names = [name for name, *_ in _semantic_cases()]
    assert len(oracle_names) == V2_ORACLE_CASES - 2
    assert len(set(oracle_names)) == len(oracle_names)


def test_oracle_rejects_analyzer_result_that_violates_the_reference_model(monkeypatch):
    original_analyze = validation_module.analyze

    def misleading_analyze(candidate, control=None):
        result = original_analyze(candidate, control)
        return replace(
            result,
            outcome=Outcome.SAFE_IN_FRAGMENT,
            reasons=(Reason.FRESH_DESTINATION,),
            borrowed_writes=(),
            collapse=None,
        )

    monkeypatch.setattr(validation_module, "analyze", misleading_analyze)
    with pytest.raises(AssertionError) as raised:
        validation_module.run_semantic_validation()
    assert str(raised.value).startswith(
        "semantic oracle mismatch: "
        "case=base(alias_destination=True,contribution=direct,"
        "reverse_operands=False,view_depth=1),"
    )


def test_oracle_rejects_forged_event_identity_and_digest(monkeypatch):
    original_analyze = validation_module.analyze

    def misleading_analyze(candidate, control=None):
        result = original_analyze(candidate, control)
        if result.borrowed_writes:
            return replace(
                result,
                candidate_digest="f" * 64,
                borrowed_writes=(replace(result.borrowed_writes[0], hop_node="n3"),),
            )
        return result

    monkeypatch.setattr(validation_module, "analyze", misleading_analyze)
    with pytest.raises(AssertionError, match="semantic oracle mismatch"):
        validation_module.run_semantic_validation()


def test_oracle_rejects_forged_collapse_fields(monkeypatch):
    original_analyze = validation_module.analyze

    def misleading_analyze(candidate, control=None):
        result = original_analyze(candidate, control)
        if result.collapse is not None:
            return replace(
                result,
                collapse=replace(result.collapse, replacement_part="hop:" + "0" * 64),
            )
        return result

    monkeypatch.setattr(validation_module, "analyze", misleading_analyze)
    with pytest.raises(AssertionError, match="semantic oracle mismatch"):
        validation_module.run_semantic_validation()


def test_oracle_accepts_exact_three_hop_collapse_and_rejects_wrong_accumulation():
    candidate, control = _micrograph(
        alias_destination=True,
        contribution="direct",
        reverse_operands=False,
        view_depth=1,
        noise=False,
        hops=3,
    )
    result = analyze(candidate, control)
    expected = _expected_micrograph(
        True, "direct", False, 1, False, hops=3
    )
    assert _matches_expected(result, expected, candidate, control) is True
    accumulation_nodes = [
        node
        for node in candidate.backward.nodes
        if node.target == "aten.add" and node.gradient_accumulation
    ]
    assert len(accumulation_nodes) == 3
    forged = replace(
        result,
        collapse=replace(
            result.collapse,
            accumulation_node=accumulation_nodes[0].id,
        ),
    )
    assert _matches_expected(forged, expected, candidate, control) is False


def test_oracle_requires_exact_unknown_detail_and_route():
    candidate, control = _micrograph(False, "direct", False, 1, False)
    candidate = _unsupported_kernel_capture(
        candidate, "mystery_call(read_role_0)"
    )
    result = analyze(candidate, control)
    expected = (
        Outcome.UNKNOWN,
        (Reason.SOURCE_PROFILE.value,),
        0,
        False,
        "kernel source profile is invalid: call mystery_call is outside the supported "
        "source subset at line 5",
    )
    assert _matches_expected(result, expected, candidate, control) is True
    assert _matches_expected(
        replace(result, unknown_detail="garbage"), expected, candidate, control
    ) is False


def test_named_ablations_are_distinct_reference_counterfactuals():
    cases = _ablation_cases()
    assert [name for name, _, _ in cases] == [
        "remove-borrowed-destination",
        "remove-direct-residual",
        "materialize-residual",
        "remove-paired-twin",
        "violate-graph-admission",
        "violate-source-effect-identity",
        "violate-view-result-identity",
    ]

    headline = _expected_micrograph(True, "direct", False, 1, False)
    assert all(expected != headline for _, _, expected in cases)
    assert [expected[:4] for _, _, expected in cases] == [
        (Outcome.SAFE_IN_FRAGMENT, (Reason.FRESH_DESTINATION.value,), 0, False),
        (Outcome.VIOLATION, (Reason.BORROWED_COTANGENT_WRITE.value,), 1, False),
        (Outcome.VIOLATION, (Reason.BORROWED_COTANGENT_WRITE.value,), 1, False),
        (Outcome.VIOLATION, (Reason.BORROWED_COTANGENT_WRITE.value,), 1, False),
        (Outcome.UNKNOWN, (Reason.GRAPH_EFFECT.value,), 0, False),
        (Outcome.UNKNOWN, (Reason.SOURCE_EFFECT.value,), 0, False),
        (Outcome.UNKNOWN, (Reason.GRAPH_EFFECT.value,), 0, False),
    ]


def test_ablations_do_not_count_the_headline_baseline():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    headline = analyze(candidate, control)

    def headline_for_every_case(candidate, control=None):
        return headline

    # Only the source/effect admission rejection remains observable; the unchanged
    # headline result must not satisfy any named semantic counterfactual.
    assert _run_ablations(analyze_fn=headline_for_every_case) == 1


def test_ablations_require_each_named_obligation_to_change():
    materialized = next(
        observation
        for name, observation, _ in _ablation_cases()
        if name == "materialize-residual"
    )
    materialized_candidate, _ = materialized
    candidate, control = _micrograph(True, "direct", False, 1, False)
    headline = analyze(candidate, control)

    def preserve_headline_for_materialized(candidate, control=None):
        if candidate == materialized_candidate:
            return headline
        return analyze(candidate, control)

    assert (
        _run_ablations(analyze_fn=preserve_headline_for_materialized)
        == ABLATION_COUNT - 1
    )


def test_mutation_denominator_and_categories_are_exact_and_input_is_unchanged():
    certificate = _certificate_fixture()
    original = copy.deepcopy(certificate)

    mutations = _certificate_mutations(certificate)
    names = [name for name, _ in mutations]

    assert certificate == original
    assert len(mutations) == V2_MUTATION_COUNT
    assert len(names) == len(set(names))
    assert sum(name.startswith("corrupt-") for name in names) == CORRUPTION_MUTATION_COUNT
    assert sum(name.startswith("closure-") for name in names) == CLOSURE_MUTATION_COUNT
    assert (
        sum(name.startswith("integrity-") for name in names)
        == EVIDENCE_INTEGRITY_MUTATION_COUNT
    )
    assert all(mutant != certificate for _, mutant in mutations)


def test_getitem_corruption_mutation_handles_multiple_gradient_outputs():
    certificate = _certificate_fixture()
    nodes = certificate["candidate"]["backward"]["nodes"]
    getitem = next(node for node in nodes if node["target"] == "operator.getitem")
    extra = copy.deepcopy(getitem)
    output = nodes.pop()
    add = nodes.pop()
    extra["id"] = f"n{len(nodes)}"
    extra["args"]["tuple"][1] = "second-output"
    add["id"] = f"n{len(nodes) + 1}"
    output["id"] = f"n{len(nodes) + 2}"
    output["args"]["tuple"][0]["tuple"][0]["ref"] = add["id"]
    nodes.extend((extra, add, output))

    mutant = dict(_corruption_mutations(certificate))["corrupt-getitem-key"]
    changed = [
        node
        for node in mutant["candidate"]["backward"]["nodes"]
        if node["target"] == "operator.getitem"
        and node["args"]["tuple"][1] == "missing-output-role"
    ]

    assert len(changed) == 1
    assert any(
        node["target"] == "operator.getitem"
        and node["args"]["tuple"][1] == "second-output"
        for node in mutant["candidate"]["backward"]["nodes"]
    )


def test_hop_binding_corruption_uses_an_existing_distinct_binding_when_all_are_bound():
    capture = {
        "backward": {
            "nodes": [
                {"id": "n0", "target": "placeholder"},
                {"id": "n1", "target": "placeholder"},
                {
                    "id": "n2",
                    "target": "triton.functional",
                    "kwargs": {
                        "kernel": "kernel",
                        "bindings": {"X_ptr": {"ref": "n0"}, "Y_ptr": {"ref": "n1"}},
                    },
                },
            ]
        },
        "kernels": [{"id": "kernel", "reads": ["X_ptr", "Y_ptr"]}],
    }

    _corrupt_hop_binding(capture)

    assert capture["backward"]["nodes"][2]["kwargs"]["bindings"] == {
        "X_ptr": {"ref": "n1"},
        "Y_ptr": {"ref": "n1"},
    }


def test_fresh_backward_allocation_accepts_the_authored_empty_like_destination():
    allocation = {
        "id": "n1",
        "target": "aten.empty_like",
        "tensor": {"alias_group": "fresh"},
    }
    capture = {
        "backward": {
            "nodes": [
                {"id": "n0", "target": "placeholder"},
                allocation,
                {
                    "id": "n2",
                    "target": "triton.functional",
                    "kwargs": {
                        "writes": ["Y_ptr"],
                        "bindings": {"Y_ptr": {"ref": "n1"}},
                    },
                },
            ]
        }
    }

    assert _fresh_backward_allocation(capture) is allocation


def test_source_closure_mutation_handles_multiple_source_files():
    certificate = _certificate_fixture()
    files = certificate["target"]["files"]
    extra = copy.deepcopy(next(entry for entry in files if entry["role"] == "source"))
    extra["packet_path"] = "external/target/tree/second.py"
    extra["target_path"] = "src/target/second.py"
    files.insert(1, extra)

    mutant = dict(_closure_mutations(certificate))["closure-target-source-snapshot"]
    changed = [
        entry
        for entry in mutant["target"]["files"]
        if entry["sha256"] == sha256_bytes(b"mutated-source-snapshot")
    ]

    assert len(changed) == 1
    assert mutant["target"]["files"][1]["sha256"] == extra["sha256"]


def test_fresh_control_alias_group_mutation_reseals_derived_analysis():
    certificate = _certificate_fixture()
    original_group = next(
        node["tensor"]["alias_group"]
        for node in certificate["control"]["backward"]["nodes"]
        if node["target"] == "aten.zeros_like"
    )
    tangent_group = next(
        node["tensor"]["alias_group"]
        for node in certificate["control"]["backward"]["nodes"]
        if isinstance(node.get("role"), dict) and node["role"].get("kind") == "tangent"
    )
    mutant = dict(_corruption_mutations(certificate))[
        "corrupt-control-fresh-alias-group"
    ]
    allocation = next(
        node
        for node in mutant["control"]["backward"]["nodes"]
        if node["target"] == "aten.zeros_like"
    )

    assert original_group != tangent_group
    assert allocation["tensor"]["alias_group"] == tangent_group
    assert mutant["analysis"]["outcome"] == "UNKNOWN"
    assert mutant["analysis"]["reasons"] == ["REGION_ALIAS"]
    assert mutant["analysis"]["unknown_detail"].startswith("REGION_ALIAS: fresh result ")
    assert " reuses storage from " in mutant["analysis"]["unknown_detail"]
    assert mutant["analysis"]["candidate_digest"] == sha256_json(mutant["candidate"])
    assert mutant["analysis"]["control_digest"] == sha256_json(mutant["control"])


def test_mutation_runner_stages_matching_isolated_packets(tmp_path):
    source_packet = tmp_path / "source-packet"
    external = source_packet / "external" / "target" / "tree"
    external.mkdir(parents=True)
    (external / "marker.txt").write_text("exact external snapshot\n", encoding="utf-8")
    reproducer = b"exact reproducer\n"
    (source_packet / "reproducer.py").write_bytes(reproducer)
    certificate_path = source_packet / "certificate.json"
    seen_packets = []

    def reject_after_sidecar_binding(path):
        raw = json.loads(path.read_text(encoding="utf-8"))
        body = {key: value for key, value in raw.items() if key != "digest"}
        assert raw["digest"] == sha256_json(body)
        assert path.read_bytes() == canonical_json_bytes(raw) + b"\n"
        for filename, section in (
            ("candidate.capture.json", "candidate"),
            ("control.capture.json", "control"),
            ("runtime.json", "runtime"),
            ("benchmark.json", "benchmark"),
            ("controls.json", "controls"),
        ):
            assert (path.parent / filename).read_bytes() == canonical_json_bytes(
                raw[section]
            ) + b"\n"
        assert (path.parent / "external" / "target" / "tree" / "marker.txt").read_text(
            encoding="utf-8"
        ) == "exact external snapshot\n"
        assert (path.parent / "reproducer.py").read_bytes() == reproducer
        assert path.parent != source_packet
        seen_packets.append(path.parent)
        raise ReplayError("intended semantic rejection")

    assert (
        validate_certificate_mutations(
            _certificate_fixture(),
            certificate_path,
            replay_fn=reject_after_sidecar_binding,
        )
        == V2_MUTATION_COUNT
    )
    assert len(seen_packets) == V2_MUTATION_COUNT
    assert len(set(seen_packets)) == V2_MUTATION_COUNT
    assert all(not packet.exists() for packet in seen_packets)


def test_target_closure_mutations_cover_specialization_and_dependency_metadata():
    certificate = _certificate_fixture()
    mutations = dict(_closure_mutations(certificate))

    specialization = mutations["closure-target-specialization"]
    assert specialization["target"]["specialization"]["expression"] == "operation(y)"
    assert specialization["candidate"] == certificate["candidate"]
    assert specialization["analysis"] == certificate["analysis"]

    source = mutations["closure-target-source-snapshot"]
    assert next(
        entry for entry in source["target"]["files"] if entry["role"] == "source"
    )["sha256"] == sha256_bytes(b"mutated-source-snapshot")

    dependency = mutations["closure-target-dependency"]
    assert next(
        entry for entry in dependency["target"]["files"] if entry["role"] == "support"
    )["sha256"] == sha256_bytes(b"mutated-dependency")
    assert dependency["candidate"] == certificate["candidate"]
    assert dependency["analysis"] == certificate["analysis"]


def test_detached_kernel_relabel_reseals_identity_hops_and_analysis():
    certificate = _certificate_fixture()
    old_id = _backward_kernel(certificate["candidate"])["id"]
    mutant = dict(_closure_mutations(certificate))["closure-detached-kernel-relabel"]

    captures = []
    new_ids = set()
    for label in ("candidate", "control"):
        raw = mutant[label]
        kernel = _backward_kernel(raw)
        new_ids.add(kernel["id"])
        assert kernel["module"] == "renamed.module"
        assert kernel["id"] != old_id
        assert kernel["id"] == sha256_json(_kernel_identity(kernel))
        assert _hop(raw)["kwargs"]["kernel"] == kernel["id"]
        assert all(
            node.get("kwargs", {}).get("kernel") != old_id
            for graph in (raw["forward"], raw["backward"])
            for node in graph["nodes"]
        )
        captures.append(Capture.from_dict(raw))

    assert len(new_ids) == 1
    expected = analyze(*captures).to_dict()
    assert mutant["analysis"] == expected
    assert mutant["analysis"]["borrowed_writes"][0]["kernel_id"] in new_ids
    assert mutant["analysis"]["candidate_digest"] == sha256_json(mutant["candidate"])
    assert mutant["analysis"]["control_digest"] == sha256_json(mutant["control"])
    for name in ("isolated", "scaled"):
        assert mutant["controls"][name]["analysis"]["borrowed_writes"][0][
            "kernel_id"
        ] in new_ids
    assert (
        mutant["analysis"]["collapse"]["replacement_part"]
        != certificate["analysis"]["collapse"]["replacement_part"]
    )


def test_producer_rejected_effect_reseals_source_identity_hops_and_unknown_analysis():
    certificate = _certificate_fixture()
    old_id = _backward_kernel(certificate["candidate"])["id"]
    mutant = dict(_closure_mutations(certificate))[
        "closure-producer-rejected-source-effect"
    ]

    new_ids = set()
    for label in ("candidate", "control"):
        raw = mutant[label]
        kernel = _backward_kernel(raw)
        new_ids.add(kernel["id"])
        assert "tl.atomic_cas" in kernel["source"]
        assert kernel["source_sha256"] == sha256_bytes(kernel["source"].encode("utf-8"))
        assert kernel["id"] == sha256_json(_kernel_identity(kernel))
        assert kernel["id"] != old_id
        assert _hop(raw)["kwargs"]["kernel"] == kernel["id"]
        with pytest.raises(InvalidArtifact, match="atomic_cas"):
            Capture.from_dict(raw)

    assert len(new_ids) == 1
    for name in ("isolated", "scaled"):
        assert mutant["controls"][name]["analysis"]["borrowed_writes"][0][
            "kernel_id"
        ] in new_ids
    assert mutant["analysis"] == {
        "outcome": "UNKNOWN",
        "reasons": ["MEMORY_MODEL"],
        "borrowed_writes": [],
        "collapse": None,
        "unknown_detail": (
            "MEMORY_MODEL: unsupported memory-model atomic effect tl.atomic_cas"
        ),
        "candidate_digest": sha256_json(mutant["candidate"]),
        "control_digest": sha256_json(mutant["control"]),
    }


def test_both_twin_unsupported_forward_recomputes_total_unknown_analysis():
    certificate = _certificate_fixture()
    mutant = dict(_closure_mutations(certificate))[
        "closure-both-twin-unsupported-forward"
    ]

    candidate = Capture.from_dict(mutant["candidate"])
    control = Capture.from_dict(mutant["control"])
    expected = analyze(candidate, control).to_dict()

    assert mutant["candidate"]["forward"] == mutant["control"]["forward"]
    assert next(
        node for node in mutant["candidate"]["forward"]["nodes"] if node["op"] == "output"
    )["target"] == "unsupported:output"
    assert mutant["analysis"] == expected
    assert expected["outcome"] == "UNKNOWN"
    assert expected["reasons"] == ["GRAPH_EFFECT"]
    assert expected["borrowed_writes"] == []
    assert expected["collapse"] is None
    assert expected["candidate_digest"] == sha256_json(mutant["candidate"])
    assert expected["control_digest"] == sha256_json(mutant["control"])


def test_evidence_integrity_mutations_are_contradictory_or_nonpositive():
    certificate = _certificate_fixture()
    mutations = dict(_evidence_integrity_mutations(certificate))
    assert all("authentic" not in name for name in mutations)

    runtime = mutations["integrity-runtime-no-witness"]["runtime"]
    for name in ("candidate", "control"):
        assert runtime[name]["compiled"]["occurrence"]["cold_ms"] == 0.0

    benchmark = mutations["integrity-benchmark-nonpositive"]["benchmark"]
    for lane in benchmark["lanes"].values():
        assert len(lane["device_ms"]) == benchmark["samples"]
        assert len(lane["wall_ms"]) == benchmark["samples"]
        assert set(lane["device_ms"]) == {0.0}
        assert set(lane["wall_ms"]) == {0.0}
        assert lane["device_median_ms"] == 0.0
        assert lane["wall_median_ms"] == 0.0
        assert lane["device_mad_ms"] == 0.0

    validation = mutations["integrity-validation-fabricated-counts"]["validation"]
    assert validation == {
        "oracle_cases": V2_ORACLE_CASES + 1,
        "oracle_passed": V2_ORACLE_CASES + 1,
        "mutation_cases": V2_MUTATION_COUNT + 1,
        "mutations_killed": V2_MUTATION_COUNT + 2,
        "ablations": ABLATION_COUNT + 1,
        "ablations_observed": ABLATION_COUNT + 1,
    }


def test_kernel_formal_alpha_renaming_preserves_ownership_projection():
    for alias_destination in (False, True):
        candidate, control = _micrograph(alias_destination, "direct", False, 1, False)
        renamed_candidate = _alpha_rename_capture(candidate)
        renamed_control = _alpha_rename_capture(control)
        original = analyze(candidate, control)
        renamed = analyze(renamed_candidate, renamed_control)

        assert _semantic_projection(renamed) == _semantic_projection(original)
        assert _backward_kernel(renamed_candidate.to_dict())["id"] != _backward_kernel(
            candidate.to_dict()
        )["id"]
        hop = _hop(renamed_candidate.to_dict())
        assert all(role.startswith("formal_") for role in hop["kwargs"]["bindings"])
        assert all(role.startswith("formal_") for role in hop["kwargs"]["writes"])


def test_target_label_and_kernel_path_relabeling_is_semantically_irrelevant():
    candidate, control = _micrograph(False, "direct", False, 1, False)
    relabeled_candidate = _relabel_capture(candidate)
    relabeled_control = _relabel_capture(control)

    assert relabeled_candidate.label != candidate.label
    relabeled_kernel = _backward_kernel(relabeled_candidate.to_dict())
    original_kernel = _backward_kernel(candidate.to_dict())
    assert relabeled_kernel["module"] == "renamed.module"
    assert relabeled_kernel["id"] != original_kernel["id"]
    assert _semantic_projection(analyze(relabeled_candidate, relabeled_control)) == _semantic_projection(
        analyze(candidate, control)
    )


def test_multiple_hops_and_fresh_allocation_forms_keep_roles_distinct():
    for allocation in ("zeros_like", "empty"):
        candidate, control = _micrograph(
            True,
            "direct",
            False,
            1,
            False,
            allocation=allocation,
            hops=3,
        )
        result = analyze(candidate, control)
        assert _semantic_projection(result) == (
            Outcome.VIOLATION,
            (Reason.BORROWED_COTANGENT_WRITE.value, Reason.CONTRIBUTION_COLLAPSE.value),
            1,
            True,
        )
        assert len(candidate.kernels) == 3
        hops = [
            node for node in candidate.backward.nodes if node.target == "triton.functional"
        ]
        assert len(hops) == 3
        assert len({node.kwargs["kernel"] for node in hops}) == 3
        assert len({kernel.runtime["jit_cache_key"] for kernel in candidate.kernels}) == 3


def test_candidate_only_safe_and_paired_collapse_are_separate_observations():
    candidate, control = _micrograph(False, "direct", False, 1, False)
    solo = analyze(candidate)
    paired = analyze(*_micrograph(True, "direct", False, 1, False))

    assert _semantic_projection(solo) == (
        Outcome.SAFE_IN_FRAGMENT,
        (Reason.FRESH_DESTINATION.value,),
        0,
        False,
    )
    assert _semantic_projection(paired) == (
        Outcome.VIOLATION,
        (Reason.BORROWED_COTANGENT_WRITE.value, Reason.CONTRIBUTION_COLLAPSE.value),
        1,
        True,
    )
    assert control is not None


def test_num_warps_changes_kernel_identity_without_changing_semantic_roles():
    candidate, control = _micrograph(True, "direct", False, 1, False)
    kernel = _backward_kernel(candidate.to_dict())
    four_warp = sha256_json(_kernel_identity(kernel))
    eight_warp_kernel = copy.deepcopy(kernel)
    eight_warp_kernel["runtime"]["config"]["num_warps"] = 8

    assert four_warp != sha256_json(_kernel_identity(eight_warp_kernel))
    assert _semantic_projection(analyze(candidate, control)) == (
        Outcome.VIOLATION,
        (Reason.BORROWED_COTANGENT_WRITE.value, Reason.CONTRIBUTION_COLLAPSE.value),
        1,
        True,
    )


def test_independent_source_packets_cover_candidate_safe_and_memory_model_boundaries():
    candidate_source = """
    @triton.jit
    def candidate_kernel(read_role_0, write_role_0):
        value = float(tl.exp(0.0))
        upper = tl.max(value, axis=0)
        for _ in tl.static_range(1):
            value = tl.where(value < upper, value, upper)
        tl.store(write_role_0, tl.load(read_role_0) + value)
    """
    candidate_packet = _source_packet(
        candidate_source,
        path="source/candidate.py",
        role="candidate-source",
        allocations=("aten.empty_like", "aten.zeros"),
    )
    _assert_source_packet(candidate_packet)
    profile = profile_triton_source(candidate_source)
    assert profile.reads == ("read_role_0",)
    assert profile.writes == ("write_role_0",)
    mutated_packet = _replace_source_packet(
        candidate_packet,
        "source/candidate.py",
        candidate_source.encode("utf-8") + b"\n",
    )
    _assert_source_packet(mutated_packet)
    assert mutated_packet != candidate_packet

    candidate, _ = _micrograph(
        False,
        "direct",
        False,
        1,
        False,
        allocation="empty",
        hops=3,
    )
    safe = analyze(candidate)
    assert _semantic_projection(safe) == (
        Outcome.SAFE_IN_FRAGMENT,
        (Reason.FRESH_DESTINATION.value,),
        0,
        False,
    )
    hops = [node for node in candidate.backward.nodes if node.target == "triton.functional"]
    kernels = {kernel.id: kernel for kernel in candidate.kernels}
    hop_kernel_ids = [node.kwargs["kernel"] for node in hops]
    assert len(hops) == 3
    assert len(set(hop_kernel_ids)) == 3
    assert len({kernels[kernel_id].runtime["jit_cache_key"] for kernel_id in hop_kernel_ids}) == 3

    first_kernel = kernels[hop_kernel_ids[0]].to_dict()
    runtime_variant = copy.deepcopy(first_kernel)
    runtime_variant["runtime"]["config"]["num_warps"] = 8
    assert runtime_variant["source_sha256"] == first_kernel["source_sha256"]
    assert sha256_json(_kernel_identity(runtime_variant)) != sha256_json(
        _kernel_identity(first_kernel)
    )

    memory_model_source = """
    @triton.jit
    def memory_model_kernel(read_role_0, write_role_0):
        while True:
            token = tl.atomic_cas(write_role_0, 0, 1)
            tl.atomic_xchg(write_role_0, token)
            tl.debug_barrier()
    """
    memory_model_packet = _source_packet(
        memory_model_source,
        path="source/memory_model.py",
        role="negative-source",
    )
    _assert_source_packet(memory_model_packet)
    assert all(
        marker in memory_model_source
        for marker in ("tl.atomic_cas", "tl.atomic_xchg", "tl.debug_barrier", "while True")
    )
    negative_candidate, _ = _micrograph(False, "direct", False, 1, False)
    unknown = analyze(
        _unsupported_kernel_capture(
            negative_candidate,
            "tl.atomic_cas(write_role_0, 0, 0)",
        )
    )
    assert unknown.outcome is Outcome.UNKNOWN
    assert unknown.reasons == (Reason.MEMORY_MODEL,)
    assert "tl.atomic_cas" in unknown.unknown_detail


def test_atomic_memory_model_and_unknown_calls_fail_closed():
    candidate, control = _micrograph(False, "direct", False, 1, False)
    for expression in ("tl.atomic_cas(write_role_0, 0, 0)", "mystery_call(read_role_0)"):
        unsupported = _unsupported_kernel_capture(candidate, expression)
        result = analyze(unsupported)
        assert result.outcome is Outcome.UNKNOWN
        expected_reason = (
            Reason.MEMORY_MODEL.value
            if expression.startswith("tl.atomic_")
            else Reason.SOURCE_PROFILE.value
        )
        assert tuple(reason.value for reason in result.reasons) == (expected_reason,)
        assert not result.borrowed_writes
        assert expression.split("(", 1)[0] in result.unknown_detail
    assert analyze(
        _unsupported_kernel_capture(candidate, "tl.atomic_cas(write_role_0, 0, 0)"), control
    ).outcome is Outcome.UNKNOWN


def test_coherent_closure_mutations_recompute_derived_analysis():
    certificate = _certificate_fixture()
    for name, mutant in _closure_mutations(certificate):
        if name == "closure-producer-rejected-source-effect":
            assert mutant["analysis"]["reasons"] == [Reason.MEMORY_MODEL.value]
            continue
        candidate = Capture.from_dict(mutant["candidate"])
        control = Capture.from_dict(mutant["control"])
        assert mutant["analysis"] == analyze(candidate, control).to_dict()
        assert mutant["analysis"]["candidate_digest"] == sha256_json(mutant["candidate"])
        assert mutant["analysis"]["control_digest"] == sha256_json(mutant["control"])


def _certificate_fixture():
    candidate, control = _micrograph(True, "direct", False, 2, False)
    backward_kernel_id = _backward_kernel(candidate.to_dict())["id"]
    checker_files = {"entrypoint.py": sha256_bytes(b"fixture checker\n")}
    source_bytes = b"fixture source"
    dependency_bytes = b"fixture dependency"
    lane = {
        "device_ms": [1.0, 2.0, 3.0],
        "wall_ms": [2.0, 3.0, 4.0],
        "device_median_ms": 2.0,
        "wall_median_ms": 3.0,
        "device_mad_ms": 1.0,
    }
    return {
        "checker": {
            "source": {
                "files": checker_files,
                "digest": sha256_json(checker_files),
            }
        },
        "target": {
            "repository": "https://example.invalid/fixture",
            "revision": sha256_bytes(b"fixture revision")[:40],
            "files": [
                {
                    "packet_path": "external/target/tree/kernel.py",
                    "target_path": "src/target/kernel.py",
                    "size": len(source_bytes),
                    "sha256": sha256_bytes(source_bytes),
                    "role": "source",
                },
                {
                    "packet_path": "external/target/tree/helper.py",
                    "target_path": "src/target/helper.py",
                    "size": len(dependency_bytes),
                    "sha256": sha256_bytes(dependency_bytes),
                    "role": "support",
                },
            ],
            "kernels": [],
            "specialization": {"expression": "operation(x)"},
        },
        "candidate": candidate.to_dict(),
        "control": control.to_dict(),
        "analysis": analyze(candidate, control).to_dict(),
        "runtime": {
            "environment": {
                "python": "3.14",
                "platform": "test",
                "torch": "test",
                "torch_git": "a" * 40,
                "triton": "test",
                "cuda_runtime": "test",
                "device": "test",
                "capability": [0, 0],
                "total_memory_bytes": 0,
                "multiprocessors": 0,
                "driver": None,
            },
            "candidate": {
                "compiled": {
                    "occurrence": {
                        "cold_ms": 1.0,
                        "generated_kernel_count": 1,
                        "ir_nodes_pre_fusion": 1,
                        "counter_delta": {"test": {"count": 1}},
                    }
                }
            },
            "control": {
                "compiled": {
                    "occurrence": {
                        "cold_ms": 1.0,
                        "generated_kernel_count": 1,
                        "ir_nodes_pre_fusion": 1,
                        "counter_delta": {"test": {"count": 1}},
                    }
                }
            },
        },
        "benchmark": {
            "samples": 3,
            "lanes": {
                "reference": copy.deepcopy(lane),
                "candidate": copy.deepcopy(lane),
                "control": copy.deepcopy(lane),
                "compiled_candidate": copy.deepcopy(lane),
                "compiled_control": copy.deepcopy(lane),
            },
        },
        "controls": {
            "isolated": {
                "analysis": {
                    "borrowed_writes": [{"kernel_id": backward_kernel_id}],
                }
            },
            "scaled": {
                "analysis": {
                    "borrowed_writes": [{"kernel_id": backward_kernel_id}],
                }
            },
            "out_of_place": {"borrowed_writes": []},
        },
        "validation": {
            "oracle_cases": V2_ORACLE_CASES,
            "oracle_passed": V2_ORACLE_CASES,
            "mutation_cases": V2_MUTATION_COUNT,
            "mutations_killed": V2_MUTATION_COUNT,
            "ablations": ABLATION_COUNT,
            "ablations_observed": ABLATION_COUNT,
        },
    }


def _source_packet(source: str, *, path: str, role: str, allocations=()):
    payload = source.encode("utf-8")
    return {
        "payloads": {path: payload},
        "files": [
            {
                "path": path,
                "size": len(payload),
                "sha256": sha256_bytes(payload),
                "role": role,
            }
        ],
        "fresh_allocations": list(allocations),
    }


def _replace_source_packet(packet, path: str, payload: bytes):
    mutated = copy.deepcopy(packet)
    mutated["payloads"][path] = payload
    entry = next(item for item in mutated["files"] if item["path"] == path)
    entry["size"] = len(payload)
    entry["sha256"] = sha256_bytes(payload)
    return mutated


def _assert_source_packet(packet):
    payloads = packet["payloads"]
    entries = packet["files"]
    assert {entry["path"] for entry in entries} == set(payloads)
    for entry in entries:
        assert entry["path"] and not entry["path"].startswith("/")
        assert ".." not in entry["path"].split("/")
        payload = payloads[entry["path"]]
        assert entry["size"] == len(payload)
        assert entry["sha256"] == sha256_bytes(payload)


def _hop(capture):
    hops = [
        node
        for node in capture["backward"]["nodes"]
        if node["target"] == "triton.functional"
    ]
    assert len(hops) == 1
    return next(iter(hops))


def _backward_kernel(capture):
    kernel_id = _hop(capture)["kwargs"]["kernel"]
    return next(kernel for kernel in capture["kernels"] if kernel["id"] == kernel_id)


def _kernel_identity(kernel):
    return {
        "module": kernel["module"],
        "qualname": kernel["qualname"],
        "source_sha256": kernel["source_sha256"],
        "arguments": kernel["argument_names"],
        "reads": kernel["reads"],
        "writes": kernel["writes"],
        "runtime": kernel["runtime"],
    }
