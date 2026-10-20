"""V2 semantic reference model, oracle cases, and counterfactuals."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Callable

from .canonical import sha256_json
from .errors import InvalidArtifact, UnsupportedCapture
from .model import Capture, Outcome, Reason
from .validation_v2_fixtures import (
    _alpha_rename_capture,
    _backward_kernel,
    _edit_capture,
    _kernel_identity,
    _micrograph,
    _node_with_target,
    _relabel_capture,
    _unsupported_kernel_capture,
)


V2_ORACLE_CASES = 48
CORRUPTION_MUTATION_COUNT = 22
CLOSURE_MUTATION_COUNT = 7
EVIDENCE_INTEGRITY_MUTATION_COUNT = 5
V2_MUTATION_COUNT = (
    CORRUPTION_MUTATION_COUNT + CLOSURE_MUTATION_COUNT + EVIDENCE_INTEGRITY_MUTATION_COUNT
)
ABLATION_COUNT = 7


def _expected_micrograph(
    alias_destination: bool,
    contribution: str,
    reverse_operands: bool,
    view_depth: int,
    noise: bool,
    *,
    paired: bool = True,
    allocation: str = "zeros_like",
    hops: int = 1,
) -> tuple[Outcome, tuple[str, ...], int, bool]:
    """Derive the expected semantic shape from storage and contribution roles.

    This reference model intentionally does not inspect ``Capture`` or call the
    ownership implementation. Views preserve the tangent storage, ``noise`` is
    disconnected from the output, and operand order cannot change contribution
    multisets. The candidate/control relation is therefore derived from the
    abstract destination and contribution identities alone.
    """
    if contribution not in {"direct", "materialized", "none"}:
        raise AssertionError(f"unknown micrograph contribution {contribution!r}")
    if view_depth not in {1, 2}:
        raise AssertionError(f"unsupported micrograph view depth {view_depth!r}")
    if allocation not in {"zeros_like", "empty"}:
        raise AssertionError(f"unsupported micrograph allocation {allocation!r}")
    if not isinstance(hops, int) or hops not in {1, 2, 3}:
        raise AssertionError(f"unsupported micrograph HOP count {hops!r}")
    if not isinstance(reverse_operands, bool) or not isinstance(noise, bool):
        raise AssertionError("micrograph axes must be boolean")

    tangent_storage = "tangent-storage"
    candidate_destination = tangent_storage if alias_destination else "fresh-storage"
    borrowed = candidate_destination == tangent_storage
    if contribution == "direct":
        control_parts = ("tangent", "kernel")
        candidate_parts = ("kernel", "kernel") if borrowed else control_parts
    elif contribution == "materialized":
        control_parts = candidate_parts = ("materialized", "kernel")
    else:
        control_parts = candidate_parts = ()
    if reverse_operands:
        control_parts = tuple(reversed(control_parts))
        candidate_parts = tuple(reversed(candidate_parts))

    collapse = paired and borrowed and contribution == "direct" and candidate_parts != control_parts
    if borrowed:
        reasons = [Reason.BORROWED_COTANGENT_WRITE.value]
        if collapse:
            reasons.append(Reason.CONTRIBUTION_COLLAPSE.value)
        return Outcome.VIOLATION, tuple(reasons), 1, collapse
    return Outcome.SAFE_IN_FRAGMENT, (Reason.FRESH_DESTINATION.value,), 0, False


def _matches_expected(
    result,
    expected: tuple,
    candidate: Capture | None = None,
    control: Capture | None = None,
) -> bool:
    expected_outcome, expected_reasons, expected_writes, expected_collapse = expected[:4]
    expected_unknown_detail = expected[4] if len(expected) > 4 else None
    projection_matches = (
        result.outcome is expected_outcome
        and tuple(reason.value for reason in result.reasons) == expected_reasons
        and len(result.borrowed_writes) == expected_writes
        and (result.collapse is not None) == expected_collapse
    )
    if not projection_matches or candidate is None:
        return projection_matches
    if (
        result.candidate_digest != candidate.digest()
        or result.control_digest != (control.digest() if control is not None else None)
        or (result.outcome is Outcome.UNKNOWN) != bool(result.unknown_detail)
        or (result.outcome is not Outcome.UNKNOWN and result.unknown_detail is not None)
    ):
        return False
    try:
        if result.outcome is Outcome.UNKNOWN:
            return (
                type(expected_unknown_detail) is str
                and result.unknown_detail == expected_unknown_detail
            )
        if _analysis_writes(result, candidate) is False:
            return False
        if expected_collapse:
            actual_collapse = (
                result.collapse.gradient_of,
                result.collapse.accumulation_node,
                result.collapse.expected_parts,
                result.collapse.actual_parts,
                result.collapse.replaced_part,
                result.collapse.replacement_part,
            )
            if actual_collapse != _collapse_expectation(candidate, control):
                return False
    except (
        AttributeError,
        IndexError,
        InvalidArtifact,
        KeyError,
        TypeError,
        ValueError,
        UnsupportedCapture,
    ):
        return False
    return True


def _analysis_writes(result, candidate: Capture) -> bool:
    """Check every reported write against its exact graph, kernel, and storage path."""
    nodes = {node.id: node for node in candidate.backward.nodes}
    kernels = {kernel.id: kernel for kernel in candidate.kernels}
    tangent = next(
        (
            node
            for node in candidate.backward.nodes
            if isinstance(node.role, dict) and node.role.get("kind") == "tangent"
        ),
        None,
    )
    if (
        tangent is None
        or tangent.tensor is None
        or not isinstance(tangent.role.get("output"), dict)
        or type(tangent.role["output"].get("index")) is not int
    ):
        return False
    tangent_group = tangent.tensor.alias_group
    tangent_name = tangent.role["output"]
    tangent_identity = f"tangent:output:{tangent_name['index']}"

    expected = []
    versions: dict[str, int] = {}
    hop_ordinals: dict[str, int] = {}
    for node in candidate.backward.nodes:
        if node.target != "triton.functional":
            continue
        kernel = kernels.get(node.kwargs.get("kernel"))
        if kernel is None:
            return False
        hop_ordinals[node.id] = len(hop_ordinals)
        bindings = node.kwargs.get("bindings", {})
        for write_role in kernel.writes:
            binding = bindings.get(write_role)
            if not isinstance(binding, dict) or set(binding) != {"ref"}:
                return False
            destination = nodes.get(binding["ref"])
            if destination is None or destination.tensor is None:
                return False
            group = destination.tensor.alias_group
            if group != tangent_group:
                continue
            before = versions.get(group, 0)
            after = before + 1
            versions[group] = after
            reads = tuple(
                role
                for role in kernel.reads
                if isinstance(bindings.get(role), dict)
                and nodes.get(bindings[role].get("ref")) is not None
                and nodes[bindings[role]["ref"]].tensor is not None
                and nodes[bindings[role]["ref"]].tensor.alias_group == group
            ) or ("<none>",)
            for read_role in reads:
                expected.append(
                    (
                        hop_ordinals[node.id],
                        write_role,
                        read_role,
                        tangent_identity,
                        node.id,
                        kernel.id,
                        before,
                        after,
                    )
                )
    expected.sort(key=lambda item: item[:3])
    actual = []
    for write in result.borrowed_writes:
        if type(write).__name__ != "BorrowedWrite":
            return False
        actual.append(
            (
                hop_ordinals.get(write.hop_node, -1),
                write.write_role,
                write.read_role,
                write.tangent,
                write.hop_node,
                write.kernel_id,
                write.before_version,
                write.after_version,
            )
        )
    actual.sort(key=lambda item: item[:3])
    return actual == expected


def _collapse_expectation(candidate: Capture, control: Capture | None):
    """Derive full Collapse fields for the bounded one-read/one-write micrographs."""
    if control is None:
        return None
    graph = candidate.backward
    nodes = {node.id: node for node in candidate.backward.nodes}
    tangent = next(node for node in graph.nodes if node.role and node.role.get("kind") == "tangent")
    tangent_name = f"tangent:output:{tangent.role['output']['index']}"
    storage_names = {
        group: f"storage:a{index}"
        for index, group in enumerate(
            dict.fromkeys(
                node.tensor.alias_group
                for node in graph.nodes
                if node.tensor and node.tensor.alias_group is not None
            )
        )
    }
    hops = [node for node in graph.nodes if node.target == "triton.functional"]
    kernels = {kernel.id: kernel for kernel in candidate.kernels}
    hop_ordinals = {hop.id: index for index, hop in enumerate(hops)}
    parts = []
    replacement = None
    for ordinal, hop in enumerate(hops):
        kernel = kernels[hop.kwargs["kernel"]]
        bindings = hop.kwargs["bindings"]
        read_role, write_role = kernel.reads[0], kernel.writes[0]
        source = nodes[bindings[read_role]["ref"]]
        destination = nodes[bindings[write_role]["ref"]]
        if source.target == "operator.getitem":
            values = source.args["tuple"]
            prior = hop_ordinals[values[0]["ref"]]
            read_parts = (parts[prior],)
            read_buffer = f"functional:{values[0]['ref']}:{values[1]}"
            read_version = 1
        elif source.tensor.alias_group == tangent.tensor.alias_group:
            read_parts = (tangent_name,)
            read_buffer = storage_names[tangent.tensor.alias_group]
            read_version = 0
        else:
            return None
        signature = {"kernel": kernel.id, "hop_ordinal": ordinal, "write_role": write_role}
        signature["write_layout"] = destination.tensor.to_dict()
        signature["write_layout"].pop("alias_group")
        read = {"buffer": read_buffer, "version": read_version, "parts": list(read_parts)}
        read["layout"] = source.tensor.to_dict()
        read["layout"].pop("alias_group")
        signature.update(
            reads={read_role: read},
            constants=hop.kwargs["constants"],
            grid=hop.kwargs["grid"],
        )
        parts.append(f"hop:{sha256_json(signature)}")
        if destination.tensor.alias_group == tangent.tensor.alias_group:
            replacement = parts[-1]

    def output(capture, parts):
        graph = capture.backward
        nodes = {node.id: node for node in capture.backward.nodes}
        hop_ordinals = {
            node.id: index
            for index, node in enumerate(
                node for node in graph.nodes if node.target == "triton.functional"
            )
        }
        tangent = next(
            node for node in graph.nodes if node.role and node.role.get("kind") == "tangent"
        )
        name = f"tangent:output:{tangent.role['output']['index']}"

        def path(node_id):
            node = nodes[node_id]
            if node.id == tangent.id:
                return (name,)
            if node.target in {"aten.view", "aten._unsafe_view", "aten.alias", "aten.detach"}:
                return path(node.args["tuple"][0]["ref"])
            values = node.args["tuple"]
            if node.target == "operator.getitem":
                return (parts[hop_ordinals[values[0]["ref"]]],)
            if node.target == "aten.add" and node.gradient_accumulation:
                return path(values[0]["ref"]) + path(values[1]["ref"])
            return None

        refs = graph.output_node().args["tuple"][0]["tuple"]
        return graph.output_roles[0]["primal"]["index"], refs[0]["ref"], path(refs[0]["ref"])

    slot, accumulation, actual = output(candidate, parts)
    control_slot, control_accumulation, expected = output(control, parts)
    if (
        slot != control_slot
        or actual is None
        or expected is None
        or Counter(actual) == Counter(expected)
    ):
        return None
    if expected.count(tangent_name) != 1 or replacement is None:
        return None
    if not any(
        node.id == accumulation and node.target == "aten.add" and node.gradient_accumulation
        for node in candidate.backward.nodes
    ):
        return None
    if not any(
        node.id == control_accumulation and node.target == "aten.add" and node.gradient_accumulation
        for node in control.backward.nodes
    ):
        return None
    substituted = Counter(expected)
    substituted[tangent_name] -= 1
    substituted[replacement] += 1
    return (
        None
        if Counter(actual) != substituted
        else (slot, accumulation, expected, actual, tangent_name, replacement)
    )


def _semantic_cases():
    cases = []
    for alias_destination in (False, True):
        for contribution in ("direct", "materialized", "none"):
            for reverse_operands in (False, True):
                for view_depth in (1, 2):
                    candidate, control = _micrograph(
                        alias_destination, contribution, reverse_operands, view_depth, False
                    )
                    cases.append(
                        (
                            f"base(alias_destination={alias_destination},"
                            f"contribution={contribution},reverse_operands={reverse_operands},"
                            f"view_depth={view_depth})",
                            candidate,
                            control,
                            _expected_micrograph(
                                alias_destination, contribution, reverse_operands, view_depth, False
                            ),
                        )
                    )
    for allocation in ("zeros_like", "empty"):
        for alias_destination in (False, True):
            candidate, control = _micrograph(
                alias_destination, "direct", False, 1, False, allocation=allocation
            )
            cases.append(
                (
                    f"fresh-allocation(alias_destination={alias_destination},"
                    f"allocation={allocation})",
                    candidate,
                    control,
                    _expected_micrograph(
                        alias_destination, "direct", False, 1, False, allocation=allocation
                    ),
                )
            )
    for alias_destination in (False, True):
        for contribution in ("direct", "none"):
            candidate, control = _micrograph(
                alias_destination, contribution, True, 1, True, allocation="empty", hops=3
            )
            cases.append(
                (
                    f"multiple-hop(alias_destination={alias_destination},"
                    f"contribution={contribution})",
                    candidate,
                    control,
                    _expected_micrograph(
                        alias_destination, contribution, True, 1, True, allocation="empty", hops=3
                    ),
                )
            )
    for alias_destination in (False, True):
        for contribution in ("direct", "materialized"):
            candidate, _ = _micrograph(alias_destination, contribution, False, 1, False)
            cases.append(
                (
                    f"candidate-only(alias_destination={alias_destination},"
                    f"contribution={contribution})",
                    candidate,
                    None,
                    _expected_micrograph(
                        alias_destination, contribution, False, 1, False, paired=False
                    ),
                )
            )
    for alias_destination in (False, True):
        for contribution in ("direct", "none"):
            candidate, control = _micrograph(alias_destination, contribution, False, 1, False)
            cases.append(
                (
                    f"relabel-invariance(alias_destination={alias_destination},"
                    f"contribution={contribution})",
                    _relabel_capture(candidate),
                    _relabel_capture(control),
                    _expected_micrograph(alias_destination, contribution, False, 1, False),
                )
            )
    for alias_destination in (False, True):
        candidate, control = _micrograph(alias_destination, "direct", False, 1, False)
        cases.append(
            (
                f"formal-alpha-renaming(alias_destination={alias_destination})",
                _alpha_rename_capture(candidate),
                _alpha_rename_capture(control),
                _expected_micrograph(alias_destination, "direct", False, 1, False),
            )
        )
    for expression, paired in (
        ("tl.atomic_cas(write_role_0, 0, 0)", False),
        ("tl.atomic_cas(write_role_0, 0, 0)", True),
        ("mystery_call(read_role_0)", False),
        ("mystery_call(read_role_0)", True),
    ):
        candidate, control = _micrograph(False, "direct", False, 1, False)
        cases.append(
            (
                f"unsupported-source(expression={expression!r},paired={paired})",
                _unsupported_kernel_capture(candidate, expression),
                control if paired else None,
                (
                    Outcome.UNKNOWN,
                    (
                        Reason.MEMORY_MODEL.value
                        if expression.startswith("tl.atomic_")
                        else Reason.SOURCE_PROFILE.value,
                    ),
                    0,
                    False,
                    (
                        "kernel source profile is invalid: MEMORY_MODEL: Triton operation tl.atomic_cas requires a thread-memory proof at line 5"
                        if expression.startswith("tl.atomic_")
                        else "kernel source profile is invalid: call mystery_call is outside the supported source subset at line 5"
                    ),
                ),
            )
        )
    if len(cases) != V2_ORACLE_CASES - 2:
        raise AssertionError(f"semantic case denominator changed: {len(cases)}")
    return cases


def _run_kernel_identity_invariance(*, analyze_fn: Callable) -> int:
    candidate, control = _micrograph(True, "direct", False, 1, False)
    result = analyze_fn(candidate, control)
    expected = _expected_micrograph(True, "direct", False, 1, False)
    if not _matches_expected(result, expected, candidate, control):
        raise AssertionError("baseline kernel identity case did not meet the reference model")
    raw_kernel = _backward_kernel(candidate.to_dict())
    four_warp_identity = sha256_json(_kernel_identity(raw_kernel))
    eight_warp = copy.deepcopy(raw_kernel)
    eight_warp["runtime"]["config"]["num_warps"] = 8
    if four_warp_identity == sha256_json(_kernel_identity(eight_warp)):
        raise AssertionError("kernel launch identity omitted num_warps")
    return 2


def _ablation_cases():
    headline_candidate, headline_control = _micrograph(True, "direct", False, 1, False)
    source_effect_error: InvalidArtifact | None = None
    try:
        _edit_capture(
            headline_candidate,
            lambda raw: _backward_kernel(raw)["source_reads"].append(
                _backward_kernel(raw)["writes"][0]
            ),
        )
    except InvalidArtifact as exc:
        source_effect_error = exc
    return (
        (
            "remove-borrowed-destination",
            _micrograph(False, "direct", False, 1, False),
            _expected_micrograph(False, "direct", False, 1, False),
        ),
        (
            "remove-direct-residual",
            _micrograph(True, "none", False, 1, False),
            _expected_micrograph(True, "none", False, 1, False),
        ),
        (
            "materialize-residual",
            _micrograph(True, "materialized", False, 1, False),
            _expected_micrograph(True, "materialized", False, 1, False),
        ),
        (
            "remove-paired-twin",
            (headline_candidate, None),
            _expected_micrograph(True, "direct", False, 1, False, paired=False),
        ),
        (
            "violate-graph-admission",
            (
                _edit_capture(
                    headline_candidate,
                    lambda raw: _node_with_target(raw, "aten.view").update(
                        target="unsupported:aten.reshape"
                    ),
                ),
                headline_control,
            ),
            (
                Outcome.UNKNOWN,
                (Reason.GRAPH_EFFECT.value,),
                0,
                False,
                "unsupported captured targets: ['unsupported:aten.reshape']",
            ),
        ),
        (
            "violate-source-effect-identity",
            source_effect_error,
            (Outcome.UNKNOWN, (Reason.SOURCE_EFFECT.value,), 0, False),
        ),
        (
            "violate-view-result-identity",
            (
                _edit_capture(
                    headline_candidate,
                    lambda raw: _node_with_target(raw, "aten.view")["tensor"].update(
                        alias_group="unrecorded-storage"
                    ),
                ),
                headline_control,
            ),
            (
                Outcome.UNKNOWN,
                (Reason.GRAPH_EFFECT.value,),
                0,
                False,
                "view node n1 result storage identity changed",
            ),
        ),
    )


def _run_ablations(*, analyze_fn: Callable) -> int:
    headline_expected = _expected_micrograph(True, "direct", False, 1, False)
    ablations = _ablation_cases()
    names = tuple(name for name, _, _ in ablations)
    if len(ablations) != ABLATION_COUNT or len(set(names)) != ABLATION_COUNT:
        raise AssertionError("ablation denominator or names changed")
    observed = 0
    for name, observation, expected in ablations:
        if expected == headline_expected:
            raise AssertionError(f"ablation {name} does not change a semantic obligation")
        if isinstance(observation, InvalidArtifact):
            matched = (
                expected[:4] == (Outcome.UNKNOWN, (Reason.SOURCE_EFFECT.value,), 0, False)
                and str(observation)
                == "kernel source signature/effects do not match recorded semantics"
            )
        elif observation is None:
            matched = False
        else:
            candidate, control = observation
            matched = _matches_expected(
                analyze_fn(candidate, control), expected, candidate, control
            )
        if matched:
            observed += 1
    return observed
