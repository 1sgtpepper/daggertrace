from __future__ import annotations

import json
import sys
from typing import Any

_SOURCE_EVIDENCE_SCHEMA = "daggertrace.aot-source-evidence/2"

def _audit_text(summary: dict[str, Any]) -> str:
    if "targets" in summary:
        targets = summary["targets"]
        sections = [
            "DaggerTrace — multi-target audit",
            "",
            f"DIRECTORY    {summary.get('directory', 'unknown')}",
            f"TARGETS      {summary.get('target_count', len(targets))}",
        ]
        sections.extend(
            f"TARGET {target}\n{_audit_text(target_summary)}"
            for target, target_summary in targets.items()
        )
        sections.extend(["", summary["status"]])
        return "\n\n".join(sections)
    target = summary["target"]
    analysis = summary["analysis"]
    target_name = summary.get("target_id", target.get("repository", "unknown target"))
    revision = target.get("revision", "unknown revision")
    lines = [
        "DaggerTrace — AOT/Triton ownership",
        "",
        f"TARGET       {target_name}@{revision}",
        f"RESULT       {analysis.get('outcome', 'UNKNOWN')}: {' + '.join(analysis.get('reasons', []))}",
    ]
    files = target.get("files")
    if isinstance(files, list):
        for source in files:
            if not isinstance(source, dict) or source.get("role") != "source":
                continue
            path = source.get("target_path")
            sha256 = source.get("sha256")
            if isinstance(path, str) and path and isinstance(sha256, str) and sha256:
                lines.append(f"SOURCE       {path}  sha256:{sha256}")
    borrowed_writes = analysis.get("borrowed_writes", [])
    if borrowed_writes:
        lines.append(f"BORROWED     {len(borrowed_writes)} source-bound write(s)")
    collapse = analysis.get("collapse")
    if collapse is not None:
        lines.append(f"COLLAPSE     {collapse}")
    replay = summary.get("replay")
    if replay is not None:
        lines.append(f"REPLAY       {replay.get('valid', False)}")
    validation = summary.get("validation")
    if isinstance(validation, dict):
        lines.append(
            "VALIDATION   "
            f"oracle={validation.get('oracle_passed', '?')}/{validation.get('oracle_cases', '?')} "
            f"mutations={validation.get('mutations_killed', '?')}/{validation.get('mutation_cases', '?')}"
        )
    lines.append("")
    certificate = summary["artifacts"].get("certificate")
    if certificate is not None:
        lines.append(f"CERTIFICATE  {certificate}")
    reproducer = summary["artifacts"].get("reproducer")
    if reproducer is not None:
        lines.append(f"REPRODUCER   {reproducer}")
    lines.extend(["", summary["status"]])
    return "\n".join(lines)


def _source_classification_text(target: str, result: dict[str, Any]) -> str:
    target_manifest = result.get("target")
    if not isinstance(target_manifest, dict):
        target_manifest = {}
    revision = target_manifest.get("revision", "unknown revision")
    outcome = result.get("outcome", "UNKNOWN")
    reason = result.get("reason")
    result_label = f"{outcome}: {reason}" if isinstance(reason, str) and reason else outcome
    lines = [
        "DaggerTrace — source-only classification",
        "",
        f"TARGET       {target}@{revision}",
        f"RESULT       {result_label}",
    ]
    files = target_manifest.get("files")
    source = files.get("source") if isinstance(files, dict) else None
    if isinstance(source, dict) and source.get("path") and source.get("sha256"):
        lines.append(f"SOURCE       {source['path']}  sha256:{source['sha256']}")
    evidence = result.get("evidence")
    reachable = evidence.get("reachable_kernels") if isinstance(evidence, dict) else None
    if isinstance(reachable, list) and reachable:
        lines.append(f"KERNELS      {', '.join(str(name) for name in reachable)}")
    detail = result.get("detail")
    if isinstance(detail, str) and detail:
        lines.append(f"DETAIL       {detail}")
    lines.extend(
        [
            "",
            "CLASSIFICATION ONLY  no AOT capture or GPU execution",
            "",
            str(result.get("status", "UNKNOWN")),
        ]
    )
    return "\n".join(lines)


def _replay_text(result: dict[str, Any]) -> str:
    if result.get("schema") == _SOURCE_EVIDENCE_SCHEMA:
        return (
            f"VALID SOURCE {result.get('source_sha256', 'unknown')}\n"
            f"{result.get('outcome', 'UNKNOWN')}: {', '.join(result.get('reasons', []))}"
        )
    return (
        f"VALID {result.get('certificate_digest', 'unknown')}\n"
        f"{result.get('outcome', 'UNKNOWN')}: {', '.join(result.get('reasons', []))}\n"
        f"borrowed writes={result.get('borrowed_writes', 0)}, "
        f"contribution collapse={result.get('contextual_collapse', False)}"
    )


def _validation_text(result: dict[str, Any]) -> str:
    return (
        f"semantic test cases {result['oracle_passed']}/{result['oracle_cases']}\n"
        f"recorded packet mutation tests {result['mutation_cases']} (not rerun)\n"
        f"ablation tests {result['ablations_observed']}/{result['ablations']}"
    )


def _reproducer_text(result: dict[str, Any]) -> str:
    lines = [f"Reproducer: {result.get('target_id', 'unknown target')}"]
    for lane in ("candidate_vs_reference", "control_vs_reference"):
        delta = result.get(lane)
        if isinstance(delta, dict) and "gx_max_abs" in delta:
            lines.append(f"{lane} max |Δgx|: {delta['gx_max_abs']:.9g}")
    if len(lines) == 1:
        lines.append(json.dumps(result, sort_keys=True))
    return "\n".join(lines)


def _emit(value: Any, as_json: bool, human: str) -> None:
    _write_text(json.dumps(value, sort_keys=True) if as_json else human, sys.stdout)


def _write_text(value: str, stream: Any) -> None:
    encoding = stream.encoding or "utf-8"
    print(value.encode(encoding, "backslashreplace").decode(encoding), file=stream)
