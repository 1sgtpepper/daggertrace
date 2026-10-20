from __future__ import annotations

from collections import Counter
from typing import Any

from .errors import InvalidArtifact, UnsupportedCapture
from .model import Analysis, Capture, Collapse, Outcome, Reason, walk_refs
from .ownership_common import (
    _ALIASES,
    _ALLOCATIONS,
    _GraphResult,
    _safe_capture_digest,
    _unknown_reason,
    _without_refs,
)
from .ownership_v2_runtime import _run_backward
from .ownership_v2_validate import _validate_capture_profile


def _find_collapse(
    candidate_capture: Capture,
    control_capture: Capture,
    candidate: _GraphResult,
    control: _GraphResult,
) -> Collapse | None:
    if set(candidate.gradients) != set(control.gradients):
        raise UnsupportedCapture("candidate/control gradient output slots differ")
    candidate_nodes = {node.id: node for node in candidate_capture.backward.nodes}
    control_nodes = {node.id: node for node in control_capture.backward.nodes}
    collapses: list[Collapse] = []
    borrowed_parts = {
        (
            candidate.borrowed_parts[(write.hop_node, write.write_role)],
            write.tangent,
        )
        for write in candidate.borrowed_writes
    }
    for slot in sorted(candidate.gradients):
        actual = candidate.gradients[slot]
        expected = control.gradients[slot]
        if Counter(actual.parts) == Counter(expected.parts):
            continue
        for lane, state, nodes in (
            ("candidate", actual, candidate_nodes),
            ("control", expected, control_nodes),
        ):
            producer = nodes.get(state.producer)
            if (
                producer is None
                or producer.target != "aten.add"
                or not producer.gradient_accumulation
            ):
                raise UnsupportedCapture(
                    f"{Reason.TWIN_RELATION.value}: {lane} gradient slot {slot} "
                    "is not produced by a marked gradient-accumulation add"
                )
        matched = None
        for hop_part, borrowed_tangent in sorted(borrowed_parts):
            substituted = Counter(expected.parts)
            if substituted[borrowed_tangent] == 0:
                continue
            substituted[borrowed_tangent] -= 1
            if not substituted[borrowed_tangent]:
                del substituted[borrowed_tangent]
            substituted[hop_part] += 1
            if Counter(actual.parts) != substituted:
                continue
            if Counter(expected.parts)[borrowed_tangent] != 1:
                raise UnsupportedCapture(
                    f"{Reason.TWIN_RELATION.value}: control gradient slot {slot} repeats {borrowed_tangent}"
                )
            matched = Collapse(
                gradient_of=slot,
                accumulation_node=actual.producer,
                expected_parts=expected.parts,
                actual_parts=actual.parts,
                replaced_part=borrowed_tangent,
                replacement_part=hop_part,
            )
            break
        if matched is None:
            raise UnsupportedCapture(
                f"{Reason.TWIN_RELATION.value}: gradient slot {slot} is not an exact tangent-to-result substitution"
            )
        collapses.append(matched)
    if len(collapses) > 1:
        raise UnsupportedCapture(f"{Reason.TWIN_RELATION.value}: multiple collapsed gradient slots")
    return collapses[0] if collapses else None


def _validate_twin(candidate: Capture, control: Capture) -> None:
    if (
        candidate.framework != control.framework
        or candidate.capture_config != control.capture_config
    ):
        raise UnsupportedCapture(
            f"{Reason.TWIN_RELATION.value}: framework/capture-configuration mismatch"
        )
    if candidate.forward.to_dict() != control.forward.to_dict():
        raise UnsupportedCapture(f"{Reason.TWIN_RELATION.value}: forward graphs differ")
    candidate_kernels = {kernel.id: kernel.to_dict() for kernel in candidate.kernels}
    control_kernels = {kernel.id: kernel.to_dict() for kernel in control.kernels}
    if candidate_kernels != control_kernels:
        raise UnsupportedCapture(f"{Reason.TWIN_RELATION.value}: source-bound kernels differ")


def _backward_twin_skeleton(capture: Capture) -> list[dict[str, Any]]:
    """Return the observable backward structure used by the paired-control rule.

    AOTAutograd may insert different view wrappers and fresh destination
    allocations for the two lanes.  Those nodes are represented by the
    ownership interpreter and deliberately omitted here.  Every other
    reachable operator remains in the topological skeleton, so an added or
    replaced computation such as a connected clone cannot hide behind the
    contribution multiset check.
    """

    graph = capture.backward
    nodes = {node.id: node for node in graph.nodes}
    reachable: set[str] = set()
    pending = list(walk_refs(graph.output_node().args))
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            continue
        node = nodes[node_id]
        reachable.add(node_id)
        pending.extend(walk_refs(node.args))
        pending.extend(walk_refs(node.kwargs))

    skeleton: list[dict[str, Any]] = []
    for node in graph.nodes:
        if node.id not in reachable:
            continue
        if node.target in _ALIASES | _ALLOCATIONS:
            continue
        tensor = None if node.tensor is None else node.tensor.to_dict()
        if tensor is not None:
            tensor.pop("alias_group")
        skeleton.append(
            {
                "op": node.op,
                "target": node.target,
                "args": _without_refs(node.args),
                "kwargs": _without_refs(node.kwargs),
                "role": node.role,
                "tensor": tensor,
                "gradient_accumulation": node.gradient_accumulation,
                "output_roles": (
                    list(graph.output_roles) if node.op == "output" else None
                ),
            }
        )
    return skeleton


def _analyze_v2(candidate: Capture, control: Capture | None = None) -> Analysis:
    """Classify v2 borrowed writes and, when paired, a contribution collapse."""

    candidate_digest = _safe_capture_digest(candidate)
    control_digest = _safe_capture_digest(control) if control is not None else None
    try:
        _validate_capture_profile(candidate)
        candidate_result = _run_backward(candidate)
        collapse = None
        if control is not None:
            _validate_capture_profile(control)
            _validate_twin(candidate, control)
            control_result = _run_backward(control)
            if control_result.borrowed_writes:
                raise UnsupportedCapture(
                    f"{Reason.TWIN_RELATION.value}: control graph contains borrowed writes"
                )
            collapse = _find_collapse(candidate, control, candidate_result, control_result)
            if _backward_twin_skeleton(candidate) != _backward_twin_skeleton(control):
                raise UnsupportedCapture(
                    f"{Reason.TWIN_RELATION.value}: normalized backward operator skeletons differ"
                )
        if candidate_result.borrowed_writes:
            reasons = [Reason.BORROWED_COTANGENT_WRITE]
            if collapse is not None:
                reasons.append(Reason.CONTRIBUTION_COLLAPSE)
            return Analysis(
                outcome=Outcome.VIOLATION,
                reasons=tuple(reasons),
                borrowed_writes=candidate_result.borrowed_writes,
                collapse=collapse,
                unknown_detail=None,
                candidate_digest=candidate_digest,
                control_digest=control_digest,
            )
        return Analysis(
            outcome=Outcome.SAFE_IN_FRAGMENT,
            reasons=(Reason.FRESH_DESTINATION,),
            borrowed_writes=(),
            collapse=None,
            unknown_detail=None,
            candidate_digest=candidate_digest,
            control_digest=control_digest,
        )
    except (
        InvalidArtifact,
        UnsupportedCapture,
        KeyError,
        TypeError,
        ValueError,
        UnicodeError,
        OverflowError,
        RecursionError,
    ) as exc:
        return Analysis(
            outcome=Outcome.UNKNOWN,
            reasons=(_unknown_reason(exc),),
            borrowed_writes=(),
            collapse=None,
            unknown_detail=str(exc),
            candidate_digest=candidate_digest,
            control_digest=control_digest,
        )
