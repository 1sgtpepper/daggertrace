from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import DaggerTraceError, TargetError
from .replay import ReplayError, replay_source_evidence
from .replay_core import _replay_certificate_with_validator
from .cli_output import _replay_text

_SOURCE_EVIDENCE_FILENAME = "source-evidence.json"
_MAX_REPLAY_PACKETS = 64
_MAX_REPLAY_ROOT_ENTRIES = 128
_MAX_REPLAY_ROOT_DEPTH = 16
_MAX_REPLAY_VISITED_ENTRIES = 1024
_MAX_REPLAY_VISITED_DIRECTORIES = 256
_REPLAY_RESERVED_DIRECTORIES = frozenset({"external", ".target-cache"})
_REPLAY_PACKET_MARKERS = frozenset({
    "certificate.json",
    _SOURCE_EVIDENCE_FILENAME,
    "candidate.capture.json",
    "control.capture.json",
    "runtime.json",
    "benchmark.json",
    "controls.json",
    "reproducer.py",
})

def replay_certificate(path: Path) -> dict[str, Any]:
    """Validate a certificate and run its target-specific checks."""

    from . import targets

    try:
        return _replay_certificate_with_validator(
            path,
            certificate_validator=targets._replay_target,
        )
    except TargetError as exc:
        raise ReplayError(str(exc)) from exc

def _replay_checked(path: Path) -> dict[str, Any]:
    result = replay_certificate(path)
    if not isinstance(result, dict) or result.get("valid") is not True:
        raise ReplayError("certificate replay did not produce a valid result")
    return result


def _replay_mutation(path: Path) -> dict[str, Any]:
    try:
        return _replay_checked(path)
    except DaggerTraceError as exc:
        raise ReplayError(str(exc)) from exc


def _source_replay_checked(path: Path) -> dict[str, Any]:
    result = replay_source_evidence(path)
    if not isinstance(result, dict) or result.get("valid") is not True:
        raise ReplayError("source-evidence replay did not produce a valid result")
    return result


def _replay_packet(path: Path) -> dict[str, Any]:
    if path.name == _SOURCE_EVIDENCE_FILENAME:
        return _source_replay_checked(path)
    return _replay_checked(path)


def _packet_paths(requested: Path) -> tuple[Path, ...]:
    if requested.is_symlink():
        raise ReplayError(f"replay packet path is a symlink: {requested}")
    if not requested.is_dir():
        return (requested,)
    packets = _discover_packet_paths(requested, depth=0)
    if not packets:
        raise ReplayError(
            f"replay packet root contains no certificate.json or {_SOURCE_EVIDENCE_FILENAME} files: {requested}"
        )
    if len(packets) > _MAX_REPLAY_PACKETS:
        raise ReplayError(
            f"replay packet root exceeds {_MAX_REPLAY_PACKETS} packets"
        )
    return tuple(sorted(packets))


def _discover_packet_paths(
    root: Path, *, depth: int, budget: list[int] | None = None
) -> list[Path]:
    if budget is None:
        budget = [0, 0]
    if depth > _MAX_REPLAY_ROOT_DEPTH:
        raise ReplayError(
            f"replay packet root exceeds {_MAX_REPLAY_ROOT_DEPTH} directory levels: {root}"
        )
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise ReplayError(f"could not enumerate replay packet root {root}: {exc}") from exc
    if len(entries) > _MAX_REPLAY_ROOT_ENTRIES:
        raise ReplayError(
            f"replay packet root exceeds {_MAX_REPLAY_ROOT_ENTRIES} entries"
        )
    budget[0] += len(entries)
    if budget[0] > _MAX_REPLAY_VISITED_ENTRIES:
        raise ReplayError(
            f"replay packet root exceeds {_MAX_REPLAY_VISITED_ENTRIES} visited entries"
        )
    budget[1] += 1
    if budget[1] > _MAX_REPLAY_VISITED_DIRECTORIES:
        raise ReplayError(
            f"replay packet root exceeds {_MAX_REPLAY_VISITED_DIRECTORIES} visited directories"
        )
    direct = _direct_packet_files(root)
    if direct:
        for child in entries:
            if child.is_symlink():
                raise ReplayError(f"replay packet root contains symlinked child {child.name}")
            if not child.is_dir() or child.name in _REPLAY_RESERVED_DIRECTORIES:
                continue
            if child.match(".*.backup-*") or child.match(".*.staging-*"):
                raise ReplayError(f"replay packet root contains transaction residue {child}")
            nested = _discover_packet_paths(child, depth=depth + 1, budget=budget)
            if nested:
                raise ReplayError("replay packet directory mixes direct and child packet layouts")
            if _looks_packet_like(child):
                raise ReplayError(f"replay packet root contains incomplete packet {child}")
        return direct

    packet_marker = any(child.name in _REPLAY_PACKET_MARKERS for child in entries)
    packets: list[Path] = []
    for child in entries:
        if child.is_symlink():
            raise ReplayError(f"replay packet root contains symlinked child {child.name}")
        if not child.is_dir() or child.name in _REPLAY_RESERVED_DIRECTORIES:
            continue
        if child.match(".*.backup-*") or child.match(".*.staging-*"):
            raise ReplayError(f"replay packet root contains transaction residue {child}")
        nested = _discover_packet_paths(child, depth=depth + 1, budget=budget)
        if nested:
            packets.extend(nested)
        elif _looks_packet_like(child):
            raise ReplayError(f"replay packet root contains incomplete packet {child}")
        if len(packets) > _MAX_REPLAY_PACKETS:
            raise ReplayError(
                f"replay packet root exceeds {_MAX_REPLAY_PACKETS} packets"
            )
    if packet_marker:
        if packets:
            raise ReplayError("replay packet directory mixes packet markers and child packets")
        raise ReplayError(f"replay packet root contains incomplete packet {root}")
    return packets


def _direct_packet_files(root: Path) -> list[Path]:
    direct: list[Path] = []
    for name in ("certificate.json", _SOURCE_EVIDENCE_FILENAME):
        path = root / name
        if path.is_symlink():
            raise ReplayError(f"replay packet root contains symlinked packet {path}")
        if path.exists() and not path.is_file():
            raise ReplayError(f"replay packet marker is not a regular file: {path}")
        if path.is_file():
            direct.append(path)
    if len(direct) > 1:
        raise ReplayError("replay packet directory contains both certificate and source evidence")
    return direct


def _looks_packet_like(root: Path) -> bool:
    try:
        return any(child.name in _REPLAY_PACKET_MARKERS for child in root.iterdir())
    except OSError as exc:
        raise ReplayError(f"could not inspect replay packet directory {root}: {exc}") from exc


def _replay_summary(
    results: list[tuple[Path, dict[str, Any]]], *, aggregate: bool = False
) -> tuple[Any, str]:
    if len(results) == 1 and not aggregate:
        result = results[0][1]
        return result, _replay_text(result)
    value = {
        "valid": all(result["valid"] for _, result in results),
        "packet_count": len(results),
        "packets": [
            {"path": path.as_posix(), "result": result} for path, result in results
        ],
    }
    sections = [f"PACKETS {len(results)}", f"VALID ROOT {value['valid']}"]
    sections.extend(
        f"PACKET {path}\n{_replay_text(result)}" for path, result in results
    )
    human = "\n\n".join(sections)
    return value, human
