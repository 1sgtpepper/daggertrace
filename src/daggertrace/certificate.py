from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .canonical import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
    write_canonical_json,
)
from .errors import InvalidArtifact
from .model import (
    Analysis,
    Capture,
)
from .model_core import _MISSING, _probe_schema, _schema_is
from .replay import ReplayError, replay_certificate
from .replay_packet import _bounded_file
from .version import (
    CAPTURE_SCHEMA,
    CERTIFICATE_SCHEMA,
    CHECKER_PROFILE,
    __version__,
)


_GENERIC_TARGET_KEYS = {"repository", "revision", "files", "kernels", "specialization"}
_FILE_KEYS = {"packet_path", "target_path", "size", "sha256", "role"}
_KERNEL_KEYS = {"module", "qualname", "file", "source_sha256"}
_ROLES = {"source", "license", "notice", "support"}
_MAX_FILE_BYTES = 8 << 20
_MAX_EXTERNAL_SOURCE_BYTES = 16 << 20
_MAX_REPRODUCER_BYTES = 4 << 10

def make_certificate(
    *,
    candidate: Capture,
    control: Capture | None,
    analysis: Analysis,
    target: dict[str, Any],
    runtime: dict[str, Any],
    benchmark: dict[str, Any],
    controls: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    if type(candidate) is not Capture or type(analysis) is not Analysis:
        raise ValueError("invalid certificate route")
    candidate_schema = object.__getattribute__(candidate, "schema")
    if type(candidate_schema) is not str:
        raise ValueError("invalid certificate route")
    if candidate_schema != CAPTURE_SCHEMA:
        raise ValueError("invalid certificate route")
    if control is not None:
        if type(control) is not Capture:
            raise ValueError("invalid certificate route")
        control_schema = object.__getattribute__(control, "schema")
        if type(control_schema) is not str or control_schema != candidate_schema:
            raise ValueError("candidate and control schemas differ")
    try:
        body = {
            "schema": CERTIFICATE_SCHEMA,
            "target": _snapshot_json(target, "certificate.target"),
            "candidate": _snapshot_json(candidate.to_dict(), "certificate.candidate"),
            "control": (
                None
                if control is None
                else _snapshot_json(control.to_dict(), "certificate.control")
            ),
            "analysis": _snapshot_json(analysis.to_dict(), "certificate.analysis"),
            "runtime": _snapshot_json(runtime, "certificate.runtime"),
            "benchmark": _snapshot_json(benchmark, "certificate.benchmark"),
            "controls": _snapshot_json(controls, "certificate.controls"),
            "validation": _snapshot_json(validation, "certificate.validation"),
        }
        body["checker"] = _checker_identity()
        return {**body, "digest": sha256_json(body)}
    except (InvalidArtifact, TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ValueError(str(exc)) from exc


def write_certificate(
    path: Path,
    certificate: dict[str, Any],
    *,
    reproducer_bytes: bytes,
) -> None:
    """Write a certificate and its supporting files as one packet.

    The online target adapter stages its external bytes before calling this function.
    The target adapter builds the manifest that independent replay validates.
    Callers must supply the reproducer bytes; the writer enforces their size limit.
    """

    path = Path(path)
    if path.name != "certificate.json":
        raise ValueError("certificate path must be named certificate.json")
    if not isinstance(certificate, dict):
        raise ValueError("unsupported certificate schema")
    if dict.__len__(certificate) > 50_000:
        raise ValueError("certificate root exceeds its structural bound")
    schema = _probe_schema(certificate)
    if schema is _MISSING:
        raise ValueError("unsupported certificate schema")
    if type(reproducer_bytes) is not bytes or len(reproducer_bytes) > _MAX_REPRODUCER_BYTES:
        raise ValueError("reproducer exceeds its size limit")
    if not _schema_is(schema, CERTIFICATE_SCHEMA):
        raise ValueError("unsupported certificate schema")
    try:
        detached = _snapshot_json(dict(dict.items(certificate)), "certificate")
        if not (
            isinstance(detached.get("target"), dict)
            and set(detached["target"]) == _GENERIC_TARGET_KEYS
        ):
            raise InvalidArtifact("certificate target is not a closed generic manifest")
        detached.pop("digest", None)
        _validate_target(detached["target"])
        if not {
            "candidate",
            "control",
            "analysis",
            "runtime",
            "benchmark",
            "controls",
            "validation",
        } <= detached.keys():
            raise InvalidArtifact("certificate sections are incomplete")
        analysis = detached["analysis"]
        expected_candidate = sha256_json(detached["candidate"])
        expected_control = None if detached["control"] is None else sha256_json(detached["control"])
        if not isinstance(analysis, dict) or (
            analysis.get("candidate_digest"), analysis.get("control_digest")
        ) != (expected_candidate, expected_control):
            raise InvalidArtifact("analysis digest relation changed")
        sealed = {**detached, "digest": sha256_json(detached)}
    except (InvalidArtifact, KeyError, TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ValueError(str(exc)) from exc
    _write_packet_transaction(path, sealed, reproducer_bytes)


def _snapshot_json(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (
        TypeError,
        ValueError,
        UnicodeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise InvalidArtifact(f"{label} is not finite canonical JSON") from exc


def _validate_target(target: Any) -> None:
    if type(target) is not dict or set(target) != _GENERIC_TARGET_KEYS:
        raise InvalidArtifact("certificate target is not a closed generic manifest")
    if any(type(target[key]) is not str or not target[key] for key in ("repository", "revision")):
        raise InvalidArtifact("certificate target identity is malformed")
    if type(target["specialization"]) is not dict:
        raise InvalidArtifact("certificate target specialization is malformed")
    files = target["files"]
    if type(files) is not list or not 0 < len(files) <= 128:
        raise InvalidArtifact("certificate target file manifest is outside its bound")
    packet_paths: set[str] = set()
    target_paths: set[str] = set()
    roles: set[str] = set()
    source_bytes = 0
    entries: dict[str, dict[str, Any]] = {}
    for item in files:
        if type(item) is not dict or set(item) != _FILE_KEYS:
            raise InvalidArtifact("certificate target file entry is malformed")
        packet = _manifest_path(item["packet_path"], packet=True)
        target_path = _manifest_path(item["target_path"], packet=False)
        if packet.casefold() in packet_paths or target_path.casefold() in target_paths:
            raise InvalidArtifact("certificate target file paths are not unique")
        size = item["size"]
        if type(size) is not int or not 0 <= size <= _MAX_FILE_BYTES:
            raise InvalidArtifact("certificate target file size is malformed")
        if not _digest(item["sha256"]) or item["role"] not in _ROLES:
            raise InvalidArtifact("certificate target file identity is malformed")
        packet_paths.add(packet.casefold())
        target_paths.add(target_path.casefold())
        roles.add(item["role"])
        entries[packet] = item
        if item["role"] == "source":
            source_bytes += size
    if (
        "source" not in roles
        or source_bytes > _MAX_EXTERNAL_SOURCE_BYTES
    ):
        raise InvalidArtifact("certificate target lacks its bounded source attribution")
    kernels = target["kernels"]
    if type(kernels) is not list or not 1 <= len(kernels) <= 64:
        raise InvalidArtifact("certificate target kernel manifest is outside its bound")
    identities: set[tuple[str, str, str]] = set()
    for item in kernels:
        if type(item) is not dict or set(item) != _KERNEL_KEYS:
            raise InvalidArtifact("certificate target kernel entry is malformed")
        if any(type(item[key]) is not str or not item[key] for key in ("module", "qualname", "file")):
            raise InvalidArtifact("certificate target kernel identity is malformed")
        file_path = _manifest_path(item["file"], packet=True)
        if file_path not in entries or entries[file_path]["role"] != "source" or not _digest(item["source_sha256"]):
            raise InvalidArtifact("certificate target kernel source is undeclared")
        identity = (item["module"], item["qualname"], item["source_sha256"])
        if identity in identities:
            raise InvalidArtifact("certificate target kernel binding is duplicated")
        identities.add(identity)


def _write_packet_transaction(
    path: Path, certificate: dict[str, Any], reproducer_bytes: bytes
) -> None:
    packet = path.parent.absolute()
    residue: Path | None = None
    try:
        _validate_packet_root(packet)
        packet = packet.resolve(strict=False)
        if packet.parent.is_dir():
            residue = next(
                (
                    entry
                    for entry in packet.parent.iterdir()
                    if entry.name.startswith(
                        (f".{packet.name}.backup-", f".{packet.name}.staging-")
                    )
                ),
                None,
            )
        if residue is not None:
            raise InvalidArtifact(f"recoverable residue: {residue}")
        packet.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{packet.name}.staging-", dir=packet.parent)
        )
    except (InvalidArtifact, OSError, RuntimeError) as exc:
        message = "certificate packet could not be written"
        if residue is not None:
            message += f"; recoverable residue: {residue}"
        raise ValueError(message) from exc

    primary_error: Exception | None = None
    cleanup_failures: list[tuple[Path, OSError]] = []
    promoted = False
    try:
        _write_staged_packet(staging, packet, certificate, reproducer_bytes)
        replay = replay_certificate(staging / "certificate.json")
        if not isinstance(replay, dict) or replay.get("valid") is not True:
            raise InvalidArtifact("certificate replay did not validate the packet")
        backup_container = _promote_packet(staging, packet)
        promoted = True
    except (InvalidArtifact, ReplayError, OSError, ValueError) as exc:
        primary_error = exc
    else:
        if backup_container is not None:
            try:
                _remove_exact(backup_container)
                if backup_container.exists() or backup_container.is_symlink():
                    raise OSError("backup residue remains")
            except OSError as exc:
                cleanup_failures.append((backup_container, exc))

    try:
        if staging.exists() or staging.is_symlink():
            _remove_exact(staging)
            if staging.exists() or staging.is_symlink():
                raise OSError("staging residue remains")
    except OSError as exc:
        cleanup_failures.append((staging, exc))
    if primary_error is None and not cleanup_failures:
        return
    residues = "; ".join(
        f"recoverable residue: {residue}: {error}"
        for residue, error in cleanup_failures
    )
    if promoted:
        raise ValueError(
            "certificate packet promotion succeeded; cleanup failed; " + residues
        ) from cleanup_failures[0][1]
    message = "certificate packet could not be written"
    if primary_error is not None:
        message += f"; primary failure: {primary_error}"
    if cleanup_failures:
        message += "; cleanup failed; " + residues
    raise ValueError(message) from (primary_error or cleanup_failures[0][1])


def _validate_packet_root(packet: Path) -> None:
    if _path_contains_symlink(packet):
        raise InvalidArtifact("certificate packet root is not a dedicated leaf")
    try:
        resolved = packet.resolve(strict=False)
        protected = {
            Path("/").resolve(),
            Path(tempfile.gettempdir()).resolve(),
            Path.home().resolve(),
            Path.cwd().resolve(),
        }
    except (OSError, RuntimeError) as exc:
        raise InvalidArtifact("certificate packet root cannot be resolved") from exc
    if resolved in protected or resolved.parent == Path("/"):
        raise InvalidArtifact("certificate packet root is not a dedicated leaf")
    if not packet.exists():
        return
    if not packet.is_dir():
        raise InvalidArtifact("certificate packet root is not a regular directory")
    try:
        entries = tuple(packet.iterdir())
    except OSError as exc:
        raise InvalidArtifact("certificate packet root cannot be inspected") from exc
    allowed = {
        "certificate.json", "candidate.capture.json", "control.capture.json", "runtime.json",
        "benchmark.json", "controls.json", "reproducer.py", "external",
    }
    for item in entries:
        if (
            item.name not in allowed
            or item.is_symlink()
            or item.name == "external" and not item.is_dir()
            or item.name != "external" and not item.is_file()
        ):
            raise InvalidArtifact("certificate packet root inventory is malformed")


def _path_contains_symlink(path: Path) -> bool:
    return any(item.is_symlink() for item in (path, *path.parents))


def _write_staged_packet(
    staging: Path,
    packet: Path,
    certificate: dict[str, Any],
    reproducer_bytes: bytes,
) -> None:
    write_canonical_json(staging / "certificate.json", certificate)
    for filename, section in (
        ("candidate.capture.json", "candidate"),
        ("control.capture.json", "control"),
        ("runtime.json", "runtime"),
        ("benchmark.json", "benchmark"),
        ("controls.json", "controls"),
    ):
        write_canonical_json(staging / filename, certificate[section])
    (staging / "reproducer.py").write_bytes(reproducer_bytes)

    for item in certificate["target"]["files"]:
        relative = Path(item["packet_path"])
        source = packet / relative
        if _path_contains_symlink(source):
            raise InvalidArtifact(f"packet external file {relative.as_posix()} traverses a symlink")
        if not source.is_file():
            raise InvalidArtifact(f"packet external file {relative.as_posix()} is absent")
        encoded = _bounded_file(
            source,
            _MAX_FILE_BYTES,
            f"external file {relative.as_posix()}",
            root=packet,
        )
        if len(encoded) != item["size"] or sha256_bytes(encoded) != item["sha256"]:
            raise InvalidArtifact(
                f"packet external file {relative.as_posix()} differs from its manifest"
            )
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)


def _promote_packet(staging: Path, packet: Path) -> Path | None:
    backup_container: Path | None = None
    backup: Path | None = None
    try:
        if packet.exists() or packet.is_symlink():
            backup_container = Path(
                tempfile.mkdtemp(prefix=f".{packet.name}.backup-", dir=packet.parent)
            )
            backup = backup_container / "previous"
            packet.replace(backup)
        staging.replace(packet)
    except OSError as promotion_error:
        if backup is not None:
            if backup.exists():
                try:
                    backup.replace(packet)
                except OSError as rollback_error:
                    raise InvalidArtifact(
                        "certificate packet promotion did not complete; could not "
                        f"restore the previous packet: {rollback_error}; "
                        f"recoverable residue: {backup_container}"
                    ) from promotion_error
            if backup_container is not None and backup_container.exists():
                try:
                    backup_container.rmdir()
                except OSError:
                    raise InvalidArtifact(
                        "certificate packet promotion did not complete; backup "
                        f"cleanup failed; recoverable residue: {backup_container}"
                    ) from promotion_error
        raise InvalidArtifact(
            f"certificate packet promotion did not complete: {promotion_error}"
        ) from promotion_error
    return backup_container


def _remove_exact(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _manifest_path(value: Any, *, packet: bool) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value or len(value.encode("utf-8")) > 1024:
        raise InvalidArtifact("certificate target path is malformed")
    parts = value.split("/")
    if (
        len(parts) > 32
        or any(not part or part in {".", ".."} or len(part.encode("utf-8")) > 255 for part in parts)
        or (packet and (len(parts) < 2 or parts[0] != "external"))
    ):
        raise InvalidArtifact("certificate target path is malformed")
    return value


def _digest(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _checker_identity() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    files = {
        path.name: sha256_file(path)
        for path in sorted(package.glob("*.py"), key=lambda item: item.name)
    }
    production_loc = sum(
        len(path.read_text(encoding="utf-8").splitlines())
        for path in sorted(package.glob("*.py"), key=lambda item: item.name)
    )
    return {
        "version": __version__,
        "profile": CHECKER_PROFILE,
        "source": {
            "files": files,
            "digest": sha256_json(files),
            "production_loc": production_loc,
        },
    }
