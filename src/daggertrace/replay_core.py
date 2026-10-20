"""Public replay functions for certificate and source-evidence packets."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .triton_protocols import (
    MEMORY_MODEL_REASON,
    MEMORY_MODEL_STATUS,
)

from .replay_packet import (
    MAX_CERTIFICATE_BYTES,
    MAX_EXTERNAL_FILES,
    MAX_EXTERNAL_FILE_BYTES,
    MAX_SOURCE_BYTES,
    MAX_STRING_BYTES,
    PROFILE,
    ReplayError,
    SCHEMA,
    SOURCE_EVIDENCE_SCHEMA,
    _TARGET_FILE_ROLES,
    _benchmark_evidence,
    _bind_runtime_environment,
    _bounded_file,
    _bounded_structure,
    _canonical,
    _checker_source,
    _closed,
    _closed_packet_inventory,
    _control_evidence,
    _integer,
    _is_digest,
    _packet_relative,
    _packet_sidecars,
    _resolve_path,
    _runtime_evidence,
    _sha,
    _sha_bytes,
    _target_manifest,
    _target_relative,
    _validation_evidence,
)
from .replay_source import _function_slice, _host_syntax_error, _source_effects
from .replay_graph import _bind_capture_kernels, _kernel_map
from .replay_v2 import (
    _analysis,
    _capture,
    _execute,
)

def replay_certificate(path: Path) -> dict[str, Any]:
    """Replay one certificate through the independent generic packet validator."""

    return _replay_certificate_with_validator(path)


def _replay_certificate_with_validator(
    path: Path,
    *,
    certificate_validator: Callable[
        [dict[str, Any], Path, dict[str, Any]], dict[str, Any] | None
    ]
    | None = None,
) -> dict[str, Any]:
    """Replay one certificate and optionally hand its canonical object to an owner."""

    try:
        packet = _resolve_path(path.parent)
        certificate_path = packet / path.name
        encoded = _bounded_file(
            certificate_path,
            MAX_CERTIFICATE_BYTES,
            "certificate.json",
            root=packet,
        )
        raw = json.loads(encoded.decode("utf-8"))
        _bounded_structure(raw, signed=True)
        if encoded != _canonical(raw) + b"\n":
            raise ReplayError("certificate bytes are not canonical JSON with one trailing newline")
        result = _replay_loaded(raw, certificate_path)
        if certificate_validator is not None:
            updates = certificate_validator(raw, certificate_path, result)
            if updates is None:
                return result
            if not isinstance(updates, dict):
                raise ReplayError("certificate validator did not return an object")
            return {**result, **updates}
        return result
    except ReplayError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        IndexError,
        RecursionError,
        SyntaxError,
        OverflowError,
    ) as exc:
        raise ReplayError(f"certificate could not be replayed: {exc}") from exc


def replay_source_evidence(path: Path) -> dict[str, Any]:
    """Validate source evidence without an AOT capture.

    This format is separate from ownership certificates. It records UNKNOWN for
    source operations outside the supported memory model; source inspection
    cannot establish SAFE_IN_FRAGMENT.
    """

    try:
        packet_root = _resolve_path(path.parent)
        evidence_path = packet_root / path.name
        encoded = _bounded_file(
            evidence_path,
            MAX_CERTIFICATE_BYTES,
            evidence_path.name,
            root=packet_root,
        )
        raw = json.loads(encoded.decode("utf-8"))
        _bounded_structure(raw)
        if encoded != _canonical(raw) + b"\n":
            raise ReplayError("source evidence bytes are not canonical JSON with one trailing newline")
        return _replay_source_evidence(raw, packet_root)
    except ReplayError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        IndexError,
        RecursionError,
        SyntaxError,
        OverflowError,
    ) as exc:
        raise ReplayError(f"source evidence could not be replayed: {exc}") from exc


def _replay_source_evidence(raw: Any, packet_root: Path) -> dict[str, Any]:
    evidence = _closed(
        raw,
        {
            "schema",
            "module",
            "qualname",
            "packet_path",
            "target_path",
            "file_sha256",
            "slice_sha256",
            "source",
            "pointer_roles",
            "analysis",
            "target",
        },
        "source evidence",
    )
    if evidence["schema"] != SOURCE_EVIDENCE_SCHEMA:
        raise ReplayError(f"unsupported source evidence schema {evidence['schema']!r}")
    for key in ("module", "qualname"):
        if not isinstance(evidence[key], str) or not evidence[key]:
            raise ReplayError(f"source evidence {key} is malformed")
    packet_path = _packet_relative(evidence["packet_path"], "source evidence.packet_path")
    if not packet_path.is_relative_to(Path("external")) or len(packet_path.parts) < 2:
        raise ReplayError("source evidence.packet_path is not an external packet path")
    target_path = _target_relative(evidence["target_path"], "source evidence.target_path")
    target = _source_target_manifest(evidence["target"])
    manifest_source = next(
        (item for item in target["files"] if item["packet_path"] == packet_path.as_posix()),
        None,
    )
    if manifest_source is None or manifest_source["role"] != "source":
        raise ReplayError("source evidence packet file is not its manifest source entry")
    if manifest_source["target_path"] != target_path:
        raise ReplayError("source evidence target path differs from its manifest entry")
    source = evidence["source"]
    if not isinstance(source, str) or not source or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ReplayError("source evidence source is malformed or oversized")
    if not _is_digest(evidence["file_sha256"]) or not _is_digest(evidence["slice_sha256"]):
        raise ReplayError("source evidence source digests are malformed")
    encoded = source.encode("utf-8")
    if _sha_bytes(encoded) != evidence["file_sha256"]:
        raise ReplayError("source evidence full-file hash does not match supplied bytes")
    if manifest_source["sha256"] != evidence["file_sha256"] or manifest_source["size"] != len(encoded):
        raise ReplayError("source evidence source bytes differ from their manifest entry")
    _closed_packet_inventory(
        packet_root,
        target,
        fixed_files={Path("source-evidence.json")},
    )
    for item in target["files"]:
        item_path = Path(item["packet_path"])
        packet_file = _bounded_file(
            packet_root / item_path,
            MAX_EXTERNAL_FILE_BYTES,
            f"source evidence target file {item_path.as_posix()}",
            root=packet_root,
        )
        if len(packet_file) != item["size"] or _sha_bytes(packet_file) != item["sha256"]:
            raise ReplayError(
                f"source evidence target file {item_path.as_posix()} differs from its manifest entry"
            )
    packet_source = _bounded_file(
        packet_root / packet_path,
        MAX_EXTERNAL_FILE_BYTES,
        f"source evidence packet file {packet_path.as_posix()}",
        root=packet_root,
    )
    if packet_source != encoded:
        raise ReplayError("source evidence packet file differs from embedded source bytes")
    try:
        function_source = _function_slice(source, evidence["qualname"], bounded=True)
    except SyntaxError as exc:
        raise _host_syntax_error(f"source evidence function {evidence['qualname']}", exc) from exc
    if _sha_bytes(function_source.encode("utf-8")) != evidence["slice_sha256"]:
        raise ReplayError("source evidence raw function slice hash does not match supplied bytes")
    pointer_roles = evidence["pointer_roles"]
    if (
        not isinstance(pointer_roles, list)
        or not pointer_roles
        or len(pointer_roles) > 64
        or len(set(pointer_roles)) != len(pointer_roles)
        or not all(isinstance(role, str) and role for role in pointer_roles)
    ):
        raise ReplayError("source evidence pointer roles are malformed")
    analysis = _closed(
        evidence["analysis"],
        {"outcome", "reasons", "borrowed_writes", "collapse", "unknown_detail"},
        "source evidence.analysis",
    )
    if analysis["outcome"] != "UNKNOWN" or analysis["reasons"] != [MEMORY_MODEL_REASON]:
        raise ReplayError(f"source evidence must persist {MEMORY_MODEL_STATUS}")
    if analysis["borrowed_writes"] != [] or analysis["collapse"] is not None:
        raise ReplayError("source evidence UNKNOWN result carries ownership state")
    if (
        not isinstance(analysis["unknown_detail"], str)
        or not analysis["unknown_detail"].startswith(f"{MEMORY_MODEL_STATUS}:")
    ):
        raise ReplayError("source evidence UNKNOWN detail is malformed")
    try:
        _source_effects(function_source, set(pointer_roles))
    except ReplayError as exc:
        if not str(exc).startswith(f"{MEMORY_MODEL_STATUS}:"):
            raise ReplayError(
                f"source evidence does not reproduce {MEMORY_MODEL_STATUS}"
            ) from exc
        if analysis["unknown_detail"] != str(exc):
            raise ReplayError("source evidence UNKNOWN detail does not match replay")
    else:
        raise ReplayError("source evidence memory-model result is not reproduced")
    return {
        "valid": True,
        "schema": SOURCE_EVIDENCE_SCHEMA,
        "outcome": "UNKNOWN",
        "reasons": [MEMORY_MODEL_REASON],
        "source_sha256": evidence["file_sha256"],
    }


def _source_target_manifest(raw: Any) -> dict[str, Any]:
    target = _closed(
        raw,
        {"repository", "revision", "files"},
        "source evidence.target",
    )
    for key in ("repository", "revision"):
        if not isinstance(target[key], str) or not target[key] or len(target[key].encode()) > MAX_STRING_BYTES:
            raise ReplayError(f"source evidence.target.{key} is malformed")
    files = target["files"]
    if not isinstance(files, list) or not 0 < len(files) <= MAX_EXTERNAL_FILES:
        raise ReplayError("source evidence.target.files is outside its bounded manifest")
    normalized_files: list[dict[str, Any]] = []
    packet_paths: set[str] = set()
    target_paths: set[str] = set()
    roles: set[str] = set()
    for index, raw_file in enumerate(files):
        item = _closed(
            raw_file,
            {"packet_path", "target_path", "size", "sha256", "role"},
            f"source evidence.target.files[{index}]",
        )
        relative = _packet_relative(
            item["packet_path"],
            f"source evidence.target.files[{index}].packet_path",
        )
        if not relative.is_relative_to(Path("external")) or len(relative.parts) < 2:
            raise ReplayError(
                f"source evidence.target.files[{index}].packet_path is not an external packet path"
            )
        target_path = _target_relative(
            item["target_path"],
            f"source evidence.target.files[{index}].target_path",
        )
        normalized_packet_path = relative.as_posix()
        folded_packet_path = normalized_packet_path.casefold()
        folded_target_path = target_path.casefold()
        if folded_packet_path in packet_paths:
            raise ReplayError("source evidence.target.files contains duplicate packet paths")
        if folded_target_path in target_paths:
            raise ReplayError("source evidence.target.files contains duplicate target paths")
        size = _integer(
            item["size"],
            f"source evidence.target.files[{index}].size",
            minimum=0,
        )
        if size > MAX_EXTERNAL_FILE_BYTES:
            raise ReplayError(f"source evidence.target.files[{index}] exceeds its byte limit")
        if not _is_digest(item["sha256"]):
            raise ReplayError(
                f"source evidence.target.files[{index}].sha256 is malformed"
            )
        if item["role"] not in _TARGET_FILE_ROLES:
            raise ReplayError(
                f"source evidence.target.files[{index}] has an unknown role"
            )
        if item["role"] in roles and item["role"] == "source":
            raise ReplayError("source evidence.target.files has multiple source entries")
        roles.add(item["role"])
        normalized = dict(item)
        normalized["packet_path"] = normalized_packet_path
        normalized["target_path"] = target_path
        normalized_files.append(normalized)
        packet_paths.add(folded_packet_path)
        target_paths.add(folded_target_path)
    if "source" not in roles or not roles & {"license", "notice"}:
        raise ReplayError("source evidence.target.files lacks source and attribution entries")
    normalized_target = dict(target)
    normalized_target["files"] = normalized_files
    return normalized_target


def _replay_loaded(raw: Any, path: Path) -> dict[str, Any]:
    certificate = _closed(
        raw,
        {
            "schema",
            "checker",
            "target",
            "candidate",
            "control",
            "analysis",
            "runtime",
            "benchmark",
            "controls",
            "validation",
            "digest",
        },
        "certificate",
    )
    digest = certificate["digest"]
    body = {key: value for key, value in certificate.items() if key != "digest"}
    if digest != _sha(body):
        raise ReplayError("certificate digest mismatch")
    if certificate["schema"] != SCHEMA:
        raise ReplayError(f"unsupported certificate schema {certificate['schema']!r}")
    target = _target_manifest(certificate["target"])
    _packet_sidecars(certificate, path, target)
    checker = _closed(certificate["checker"], {"version", "profile", "source"}, "checker")
    if checker["version"] != "0.4.0" or checker["profile"] != PROFILE:
        raise ReplayError(
            f"unsupported checker identity {checker['version']!r}/{checker['profile']!r}"
        )
    _checker_source(checker["source"])
    candidate = _capture(certificate["candidate"], "candidate")
    control = None if certificate["control"] is None else _capture(certificate["control"], "control")
    _bind_capture_kernels(candidate, target, path.parent, "candidate")
    if control is not None:
        _bind_capture_kernels(control, target, path.parent, "control")
        if (
            _canonical(candidate["framework"]) != _canonical(control["framework"])
            or _canonical(candidate["capture_config"]) != _canonical(control["capture_config"])
        ):
            raise ReplayError("candidate/control framework or capture configuration differ")
        if _canonical(candidate["forward"]) != _canonical(control["forward"]):
            raise ReplayError("candidate/control forward graphs differ")
        if _canonical(_kernel_map(candidate)) != _canonical(_kernel_map(control)):
            raise ReplayError("candidate/control kernel identities differ")
    candidate_result = _execute(candidate)
    control_result = None if control is None else _execute(control)
    derived = _analysis(candidate, control, candidate_result, control_result)
    if _canonical(certificate["analysis"]) != _canonical(derived):
        raise ReplayError("recorded ownership analysis does not match literal replay")
    runtime_environment = _runtime_evidence(certificate["runtime"])
    _bind_runtime_environment(runtime_environment, candidate, control)
    _benchmark_evidence(certificate["benchmark"])
    _control_evidence(certificate["controls"])
    _validation_evidence(certificate["validation"])
    return {
        "valid": True,
        "certificate_digest": digest,
        "outcome": derived["outcome"],
        "reasons": derived["reasons"],
        "borrowed_writes": len(derived["borrowed_writes"]),
        "contextual_collapse": derived["collapse"] is not None,
    }
