"""Bounded packet, path, and canonical JSON primitives for replay."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import statistics
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


SCHEMA = "daggertrace.cotangent-ownership/2"
CAPTURE_SCHEMA = "daggertrace.aot-capture/2"
SOURCE_EVIDENCE_SCHEMA = "daggertrace.aot-source-evidence/2"
PROFILE = "torch-2.12-triton-hop-whole-storage/2"
SUPPORTED_TORCH = "2.12.1+cu130"
SUPPORTED_TORCH_GIT = "7269437d655783a26cba32aa88195b741ff496aa"
SUPPORTED_TRITON = "3.7.1"
EXPECTED_CAPTURE_CONFIG = {
    "fullgraph": True,
    "dynamic": False,
    "stage": "post-dynamo-pre-inductor",
}
MAX_CERTIFICATE_BYTES = 8 << 20
MAX_STRUCTURE_ITEMS = 100_000
MAX_STRUCTURE_DEPTH = 64
MAX_STRING_BYTES = 1 << 20
MAX_EXTERNAL_FILES = 128
MAX_EXTERNAL_FILE_BYTES = 8 << 20
MAX_PACKET_BYTES = 64 << 20
MAX_PATH_BYTES = 1024
MAX_PATH_COMPONENTS = 32
MAX_SOURCE_BYTES = 16 << 20
MAX_KERNELS = 64
MAX_DERIVED_PARTS = 4096
CONFIG_COUNT_MAX = 32
_ALLOCATION_DTYPES = {
    "bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32",
    "int64", "float16", "bfloat16", "float32", "float64", "complex64", "complex128",
}
_COMPATIBLE_CHECKER_TUPLES = {
    (SCHEMA, "0.4.0", PROFILE, "0b055a280bed62a00cb1a6861a55a42e685fae203c832849c05336929a7fe4dd", 8974),
}
_TARGET_FILE_ROLES = {"source", "license", "notice", "support"}
_FIXED_FILES = {
    Path("certificate.json"),
    Path("candidate.capture.json"),
    Path("control.capture.json"),
    Path("runtime.json"),
    Path("benchmark.json"),
    Path("controls.json"),
    Path("reproducer.py"),
}
_ALIASES = {"aten.view", "aten._unsafe_view", "aten.alias", "aten.detach"}
_ALLOCATIONS = {"aten.zeros_like", "aten.empty_like", "aten.empty", "aten.zeros"}
_ALLOCATION_KWARGS = {"dtype", "layout", "device", "pin_memory", "memory_format"}
_PURE = {"aten.mul", "aten.sum"}
_STORAGE_ROOT_TARGETS = _ALLOCATIONS | _PURE | {
    "aten.clone",
    "aten.add",
    "operator.getitem",
}
_HOP_WRITE_RESULT_FIELDS = ("shape", "stride", "dtype", "device", "storage_offset")
_CALL_TARGETS = _ALIASES | _ALLOCATIONS | _PURE | {
    "aten.clone",
    "aten.add",
    "operator.getitem",
    "triton.functional",
}


class ReplayError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _State:
    buffer: str
    version: int
    parts: tuple[str, ...]
    tangent: str | None
    producer: str

def _target_manifest(raw: Any) -> dict[str, Any]:
    target = _closed(
        raw,
        {"repository", "revision", "files", "kernels", "specialization"},
        "target",
    )
    for key in ("repository", "revision"):
        if not isinstance(target[key], str) or not target[key] or len(target[key].encode()) > MAX_STRING_BYTES:
            raise ReplayError(f"target.{key} is malformed")
    if not isinstance(target["specialization"], dict):
        raise ReplayError("target.specialization must be a JSON object")
    _bounded_structure(target["specialization"])

    files = target["files"]
    if not isinstance(files, list) or not 0 < len(files) <= MAX_EXTERNAL_FILES:
        raise ReplayError("target.files is outside its bounded manifest")
    manifest_files: dict[str, dict[str, Any]] = {}
    normalized_files: list[dict[str, Any]] = []
    packet_paths: set[str] = set()
    target_paths: set[str] = set()
    for index, raw_file in enumerate(files):
        item = _closed(
            raw_file,
            {"packet_path", "target_path", "size", "sha256", "role"},
            f"target.files[{index}]",
        )
        relative = _packet_relative(item["packet_path"], f"target.files[{index}].packet_path")
        if not relative.is_relative_to(Path("external")) or len(relative.parts) < 2:
            raise ReplayError(f"target.files[{index}].packet_path is not an external packet path")
        target_path = _target_relative(item["target_path"], f"target.files[{index}].target_path")
        normalized_packet_path = relative.as_posix()
        folded = normalized_packet_path.casefold()
        if folded in packet_paths:
            raise ReplayError(f"target.files contains duplicate path {relative.as_posix()}")
        folded_target_path = target_path.casefold()
        if folded_target_path in target_paths:
            raise ReplayError(f"target.files contains duplicate target path {target_path}")
        size = _integer(item["size"], f"target.files[{index}].size", minimum=0)
        if size > MAX_EXTERNAL_FILE_BYTES:
            raise ReplayError(f"target.files[{index}] exceeds its byte limit")
        if not _is_digest(item["sha256"]):
            raise ReplayError(f"target.files[{index}].sha256 is malformed")
        if item["role"] not in _TARGET_FILE_ROLES:
            raise ReplayError(f"target.files[{index}] has an unknown role")
        item = dict(item)
        item["packet_path"] = normalized_packet_path
        item["target_path"] = target_path
        manifest_files[normalized_packet_path] = item
        normalized_files.append(item)
        packet_paths.add(folded)
        target_paths.add(folded_target_path)

    kernels = target["kernels"]
    if not isinstance(kernels, list) or not 1 <= len(kernels) <= MAX_KERNELS:
        raise ReplayError("target.kernels is outside its bounded manifest")
    seen_bindings: set[tuple[str, str, str]] = set()
    for index, raw_kernel in enumerate(kernels):
        item = _closed(
            raw_kernel,
            {"module", "qualname", "file", "source_sha256"},
            f"target.kernels[{index}]",
        )
        for key in ("module", "qualname", "file"):
            if not isinstance(item[key], str) or not item[key]:
                raise ReplayError(f"target.kernels[{index}].{key} is malformed")
        file_path = _packet_relative(item["file"], f"target.kernels[{index}].file")
        if file_path.as_posix() not in manifest_files:
            raise ReplayError(f"target.kernels[{index}] references an undeclared file")
        if manifest_files[file_path.as_posix()]["role"] != "source":
            raise ReplayError(f"target.kernels[{index}] does not reference a source file")
        if not _is_digest(item["source_sha256"]):
            raise ReplayError(f"target.kernels[{index}].source_sha256 is malformed")
        identity = (item["module"], item["qualname"], item["source_sha256"])
        if identity in seen_bindings:
            raise ReplayError("target.kernels contains duplicate bindings")
        seen_bindings.add(identity)
    source_bytes = sum(
        item["size"] for item in manifest_files.values() if item["role"] == "source"
    )
    if source_bytes > MAX_SOURCE_BYTES:
        raise ReplayError("target source manifest exceeds its aggregate byte limit")
    normalized_target = dict(target)
    normalized_target["files"] = normalized_files
    return normalized_target


def _packet_relative(raw: Any, label: str) -> Path:
    parts = _relative_path_parts(
        raw,
        label,
        f"{label} is not a normalized relative path",
        component_limit=255,
    )
    return Path(*parts)


def _target_relative(raw: Any, label: str) -> str:
    parts = _relative_path_parts(
        raw,
        label,
        f"{label} is not valid provenance",
        component_limit=None,
    )
    return "/".join(parts)


def _relative_path_parts(
    raw: Any,
    label: str,
    invalid: str,
    *,
    component_limit: int | None,
) -> tuple[str, ...]:
    if not isinstance(raw, str) or not raw or "\\" in raw or "\x00" in raw:
        raise ReplayError(invalid)
    try:
        encoded = raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ReplayError(invalid) from exc
    if len(encoded) > MAX_PATH_BYTES:
        raise ReplayError(f"{label} exceeds its byte limit")
    path = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    parts = path.parts
    if (
        path.is_absolute()
        or path.as_posix() != raw
        or windows.drive
        or windows.root
        or not parts
        or len(parts) > MAX_PATH_COMPONENTS
        or any(
            part in {"", ".", ".."}
            or PureWindowsPath(part).drive
            or component_limit is not None and len(part.encode("utf-8")) > component_limit
            for part in parts
        )
    ):
        raise ReplayError(invalid)
    return parts

def _checker_source(raw: Any) -> None:
    source = _closed(raw, {"files", "digest", "production_loc"}, "checker.source")
    files = source["files"]
    if not isinstance(files, dict) or not files or not all(
        isinstance(name, str) and isinstance(digest, str) and _is_digest(digest)
        for name, digest in files.items()
    ):
        raise ReplayError("checker source manifest is malformed")
    if source["digest"] != _sha(files):
        raise ReplayError("checker source manifest digest mismatch")
    package = Path(__file__).resolve().parent
    local_paths = sorted(package.glob("*.py"), key=lambda item: item.name)
    local_files = {path.name: _sha_bytes(path.read_bytes()) for path in local_paths}
    local_loc = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in local_paths)
    production_loc = _integer(
        source["production_loc"],
        "checker.source.production_loc",
        minimum=0,
    )
    identity = (SCHEMA, "0.4.0", PROFILE)
    if files == local_files:
        if production_loc != local_loc:
            raise ReplayError("checker production LOC does not match the source manifest")
        return
    if (*identity, source["digest"], production_loc) in _COMPATIBLE_CHECKER_TUPLES:
        return
    raise ReplayError("certificate checker source identity is unsupported")


def _packet_sidecars(
    certificate: dict[str, Any], certificate_path: Path, target: dict[str, Any] | None = None
) -> None:
    packet = _resolve_path(certificate_path.parent)
    if certificate_path.name != "certificate.json":
        raise ReplayError(f"packet certificate filename changed: {certificate_path.name}")
    if target is None:
        target = _target_manifest(certificate["target"])
    _closed_packet_inventory(packet, target)
    sections = {
        "candidate.capture.json": "candidate",
        "control.capture.json": "control",
        "runtime.json": "runtime",
        "benchmark.json": "benchmark",
        "controls.json": "controls",
    }
    for filename, section in sections.items():
        encoded = _bounded_file(packet / filename, MAX_CERTIFICATE_BYTES, filename, root=packet)
        if encoded != _canonical(certificate[section]) + b"\n":
            raise ReplayError(f"packet sidecar {filename} differs from its embedded certificate section")
    # Reproducer bytes are target-owned supporting evidence, not generic semantics.
    _bounded_file(packet / "reproducer.py", 4 << 10, "reproducer.py", root=packet)

    for item in target["files"]:
        relative = item["packet_path"]
        encoded = _bounded_file(
            packet / relative,
            MAX_EXTERNAL_FILE_BYTES,
            f"external file {relative}",
            root=packet,
        )
        if len(encoded) != item["size"] or _sha_bytes(encoded) != item["sha256"]:
            raise ReplayError(f"external file {relative} differs from its manifest entry")


def _closed_packet_inventory(
    packet: Path,
    target: dict[str, Any],
    *,
    fixed_files: set[Path] = _FIXED_FILES,
) -> None:
    expected_files = set(fixed_files)
    expected_files.update(Path(item["packet_path"]) for item in target["files"])
    if len(expected_files) > MAX_EXTERNAL_FILES + len(fixed_files):
        raise ReplayError("packet inventory exceeds its entry limit")
    expected_directories = {Path()}
    for relative in expected_files:
        expected_directories.update(relative.parents)
    expected_directories.discard(Path("."))
    expected_directories = {item for item in expected_directories if item != Path()}
    children: dict[Path, dict[str, str]] = {directory: {} for directory in expected_directories | {Path()}}
    for directory in expected_directories:
        children[directory.parent][directory.name] = "directory"
    for file in expected_files:
        children[file.parent][file.name] = "regular file"

    for relative in sorted(expected_directories | {Path()}, key=lambda item: item.as_posix()):
        directory = packet / relative
        expected = children[relative]
        seen: set[str] = set()
        try:
            metadata = os.lstat(_filesystem_path(directory))
            if not stat.S_ISDIR(metadata.st_mode):
                raise ReplayError(f"packet inventory path {relative.as_posix() or '.'} is not a directory")
            with os.scandir(_filesystem_path(directory)) as entries:
                for entry in entries:
                    name = _filesystem_name(entry.name)
                    entry_relative = relative / name
                    expected_kind = expected.get(name)
                    if expected_kind is None:
                        raise ReplayError(
                            f"packet inventory contains unexpected entry {entry_relative.as_posix()}"
                        )
                    mode = entry.stat(follow_symlinks=False).st_mode
                    valid_kind = (
                        stat.S_ISDIR(mode) if expected_kind == "directory" else stat.S_ISREG(mode)
                    )
                    if not valid_kind:
                        raise ReplayError(
                            f"packet inventory entry {entry_relative.as_posix()} is not a {expected_kind}"
                        )
                    seen.add(name)
        except ReplayError:
            raise
        except OSError as exc:
            label = relative.as_posix() if relative.parts else "."
            raise ReplayError(f"packet inventory directory {label} is unreadable") from exc
        missing = sorted(set(expected) - seen)
        if missing:
            prefix = relative.as_posix() if relative.parts else "."
            raise ReplayError(f"packet inventory directory {prefix} is missing {', '.join(missing)}")

    aggregate = 0
    for relative in expected_files:
        try:
            metadata = os.lstat(_filesystem_path(packet / relative))
        except OSError as exc:
            raise ReplayError(f"packet inventory file {relative.as_posix()} is unreadable") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise ReplayError(f"packet inventory entry {relative.as_posix()} is not a regular file")
        aggregate += metadata.st_size
        if aggregate > MAX_PACKET_BYTES:
            raise ReplayError("packet inventory exceeds its aggregate byte limit")


def _bounded_file(path: Path, limit: int, label: str, *, root: Path) -> bytes:
    root = _resolve_path(root)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ReplayError(f"packet file {label} is outside the packet root") from exc
    if not relative.parts or ".." in relative.parts:
        raise ReplayError(f"packet file {label} is outside the packet root")
    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = common_flags | getattr(os, "O_DIRECTORY", 0)
    file_flags = common_flags | getattr(os, "O_NONBLOCK", 0)
    directories: list[int] = []
    descriptor: int | None = None
    try:
        current = os.open(_filesystem_path(root), directory_flags)
        directories.append(current)
        for component in relative.parts[:-1]:
            current = os.open(_filesystem_path(component), directory_flags, dir_fd=current)
            directories.append(current)
        descriptor = os.open(_filesystem_path(relative.parts[-1]), file_flags, dir_fd=current)
    except OSError as exc:
        for directory in reversed(directories):
            os.close(directory)
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ReplayError(f"packet file {label} traverses a symlink") from exc
        raise ReplayError(f"packet file {label} is absent or unreadable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ReplayError(f"packet file {label} is not a regular file")
        if metadata.st_size > limit:
            raise ReplayError(f"packet file {label} exceeds its byte limit")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            encoded = handle.read(limit + 1)
    except OSError as exc:
        raise ReplayError(f"packet file {label} is unreadable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for directory in reversed(directories):
            os.close(directory)
    if len(encoded) > limit:
        raise ReplayError(f"packet file {label} exceeds its byte limit")
    return encoded


def _resolve_path(path: Path) -> Path:
    try:
        return path.resolve()
    except UnicodeEncodeError:
        return path.absolute()
    except RuntimeError as exc:
        raise ReplayError("packet path cannot be resolved") from exc


def _filesystem_path(path: Path | str) -> str | bytes:
    value = os.fspath(path)
    try:
        os.fsencode(value)
    except UnicodeEncodeError:
        if os.name == "posix":
            return value.encode("utf-8", "surrogateescape")
        raise
    return value


def _filesystem_name(name: str | bytes) -> str:
    if isinstance(name, str):
        if not any("\udc80" <= character <= "\udcff" for character in name):
            return name
        try:
            return name.encode("ascii", "surrogateescape").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return name
    try:
        return name.decode("utf-8")
    except UnicodeDecodeError:
        return os.fsdecode(name)

def _runtime_evidence(raw: Any) -> dict[str, Any]:
    runtime = _closed(raw, {"environment", "candidate", "control"}, "runtime")
    environment = _environment(runtime["environment"])
    _runtime_lane(runtime["candidate"], "runtime.candidate", required=True)
    _runtime_lane(runtime["control"], "runtime.control", required=False)
    return environment


def _bind_runtime_environment(
    environment: dict[str, Any],
    candidate: dict[str, Any],
    control: dict[str, Any] | None,
) -> None:
    keys = ("python", "torch", "torch_git", "triton", "cuda_runtime")
    for lane, capture in (("candidate", candidate), ("control", control)):
        if capture is None:
            continue
        if any(environment[key] != capture["framework"][key] for key in keys):
            raise ReplayError(f"runtime environment differs from the {lane} capture framework")


def _runtime_lane(raw: Any, label: str, *, required: bool) -> None:
    if raw is None:
        if required:
            raise ReplayError(f"{label} is required")
        return
    lane = _closed(raw, {"compiled"}, label)
    compiled = lane["compiled"]
    if compiled is None:
        return
    compiled = _closed(compiled, {"occurrence"}, f"{label}.compiled")
    occurrence = _closed(
        compiled["occurrence"],
        {"cold_ms", "generated_kernel_count", "ir_nodes_pre_fusion", "counter_delta"},
        f"{label}.compiled.occurrence",
    )
    _number(occurrence["cold_ms"], f"{label}.compiled.occurrence.cold_ms", positive=True)
    _integer(
        occurrence["generated_kernel_count"],
        f"{label}.compiled.occurrence.generated_kernel_count",
        minimum=0,
        maximum=MAX_KERNELS,
    )
    _integer(
        occurrence["ir_nodes_pre_fusion"],
        f"{label}.compiled.occurrence.ir_nodes_pre_fusion",
        minimum=0,
        maximum=MAX_STRUCTURE_ITEMS,
    )
    counters = occurrence["counter_delta"]
    if not isinstance(counters, dict) or len(counters) > 64 or not all(
        isinstance(key, str) and key and isinstance(value, dict) and len(value) <= 64
        for key, value in counters.items()
    ):
        raise ReplayError(f"{label}.compiled.occurrence.counter_delta is malformed")
    for namespace, values in counters.items():
        for counter, value in values.items():
            if not isinstance(counter, str) or not counter:
                raise ReplayError(
                    f"{label}.compiled.occurrence.{namespace} counter name is malformed"
                )
            _integer(
                value,
                f"{label}.compiled.occurrence.{namespace}.{counter}",
                minimum=0,
            )


def _benchmark_evidence(raw: Any):
    benchmark = _closed(raw, {"warmups", "samples", "lanes", "claim"}, "benchmark")
    if benchmark["claim"] != "supporting evidence only":
        raise ReplayError("benchmark claim boundary changed")
    _integer(benchmark["warmups"], "benchmark.warmups", minimum=1, maximum=10)
    samples = _integer(benchmark["samples"], "benchmark.samples", minimum=3, maximum=50)
    lanes = benchmark["lanes"]
    if not isinstance(lanes, dict) or not lanes or len(lanes) > 32 or not all(isinstance(name, str) and name for name in lanes):
        raise ReplayError("benchmark sample record is malformed")
    for name, raw_lane in lanes.items():
        lane = _closed(raw_lane, {"device_ms", "wall_ms", "device_median_ms", "wall_median_ms", "device_mad_ms"}, f"benchmark.{name}")
        if not isinstance(lane["device_ms"], list) or not isinstance(lane["wall_ms"], list) or len(lane["device_ms"]) != samples or len(lane["wall_ms"]) != samples:
            raise ReplayError("benchmark raw sample count mismatch")
        device = [_number(value, f"benchmark.{name}.device_ms", positive=True) for value in lane["device_ms"]]
        wall = [_number(value, f"benchmark.{name}.wall_ms", positive=True) for value in lane["wall_ms"]]
        median = statistics.median(device)
        mad = statistics.median(abs(value - median) for value in device)
        recorded = (
            _number(lane["device_median_ms"], f"benchmark.{name}.device_median_ms", positive=True),
            _number(lane["wall_median_ms"], f"benchmark.{name}.wall_median_ms", positive=True),
            _number(lane["device_mad_ms"], f"benchmark.{name}.device_mad_ms"),
        )
        expected = (median, statistics.median(wall), mad)
        if any(not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12) for left, right in zip(recorded, expected)):
            raise ReplayError(f"benchmark {name} derived statistics changed")


def _control_evidence(raw: Any):
    if not isinstance(raw, dict) or len(raw) > 32 or not all(isinstance(key, str) and key for key in raw):
        raise ReplayError("controls evidence is malformed")
    for name, value in raw.items():
        if isinstance(value, dict) and set(value) == {"outcome", "reasons", "borrowed_writes", "collapse", "unknown_detail", "candidate_digest", "control_digest"}:
            _analysis_record(value, f"controls.{name}")
            continue
        item = _closed(value, {"analysis"}, f"controls.{name}")
        _analysis_record(item["analysis"], f"controls.{name}.analysis")


def _validation_evidence(raw: Any):
    expected = {"oracle_cases", "oracle_passed", "mutation_cases", "mutations_killed", "ablations", "ablations_observed"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ReplayError("validation summary fields changed")
    for key in expected:
        _integer(raw[key], f"validation.{key}", minimum=0)
    if raw["oracle_passed"] > raw["oracle_cases"] or raw["mutations_killed"] > raw["mutation_cases"] or raw["ablations_observed"] > raw["ablations"]:
        raise ReplayError("validation summary exceeds its denominators")


def _environment(raw: Any) -> dict[str, Any]:
    environment = _closed(raw, {"python", "platform", "torch", "torch_git", "triton", "cuda_runtime", "device", "capability", "total_memory_bytes", "multiprocessors", "driver"}, "runtime.environment")
    for key in ("python", "platform", "torch", "torch_git", "triton", "cuda_runtime", "device"):
        if not isinstance(environment[key], str) or not environment[key]:
            raise ReplayError(f"runtime.environment.{key} is malformed")
    if (
        environment["torch"] != SUPPORTED_TORCH
        or environment["torch_git"] != SUPPORTED_TORCH_GIT
        or environment["triton"] != SUPPORTED_TRITON
    ):
        raise ReplayError("runtime environment is outside the versioned framework profile")
    capability = environment["capability"]
    if not isinstance(capability, list) or not capability:
        raise ReplayError("runtime environment capability is malformed")
    for index, value in enumerate(capability):
        _integer(value, f"runtime.environment.capability[{index}]", minimum=0)
    _integer(environment["total_memory_bytes"], "runtime.environment.total_memory_bytes", minimum=0)
    _integer(environment["multiprocessors"], "runtime.environment.multiprocessors", minimum=0)
    if environment["driver"] is not None and not isinstance(environment["driver"], str):
        raise ReplayError("runtime.environment.driver is malformed")
    return environment


def _analysis_record(raw: Any, label: str) -> None:
    analysis = _closed(raw, {"outcome", "reasons", "borrowed_writes", "collapse", "unknown_detail", "candidate_digest", "control_digest"}, label)
    if analysis["outcome"] not in {"VIOLATION", "SAFE_IN_FRAGMENT"}:
        raise ReplayError(f"{label}.outcome changed")
    if not isinstance(analysis["reasons"], list) or not all(isinstance(reason, str) for reason in analysis["reasons"]) or len(set(analysis["reasons"])) != len(analysis["reasons"]):
        raise ReplayError(f"{label}.reasons is malformed")
    if not isinstance(analysis["borrowed_writes"], list):
        raise ReplayError(f"{label}.borrowed_writes is malformed")
    for write in analysis["borrowed_writes"]:
        item = _closed(write, {"tangent", "hop_node", "read_role", "write_role", "kernel_id", "before_version", "after_version"}, f"{label}.borrowed_write")
        if not all(isinstance(item[key], str) and item[key] for key in ("tangent", "hop_node", "read_role", "write_role")) or not _is_digest(item["kernel_id"]):
            raise ReplayError(f"{label}.borrowed_write is malformed")
        _integer(item["before_version"], f"{label}.borrowed_write.before_version", minimum=0)
        _integer(item["after_version"], f"{label}.borrowed_write.after_version", minimum=0)
        if item["after_version"] <= item["before_version"]:
            raise ReplayError(f"{label}.borrowed_write versions do not advance")
    collapse = analysis["collapse"]
    if collapse is not None:
        collapse = _closed(
            collapse,
            {
                "gradient_of",
                "accumulation_node",
                "expected_parts",
                "actual_parts",
                "replaced_part",
                "replacement_part",
            },
            f"{label}.collapse",
        )
        _integer(collapse["gradient_of"], f"{label}.collapse.gradient_of", minimum=0)
        for key in ("accumulation_node", "replaced_part", "replacement_part"):
            if not isinstance(collapse[key], str) or not collapse[key]:
                raise ReplayError(f"{label}.collapse.{key} is malformed")
        for key in ("expected_parts", "actual_parts"):
            parts = collapse[key]
            if not isinstance(parts, list) or not parts or not all(
                isinstance(part, str) and part for part in parts
            ):
                raise ReplayError(f"{label}.collapse.{key} is malformed")
    if analysis["unknown_detail"] is not None:
        raise ReplayError(
            f"{label}.unknown_detail is not valid in an ownership certificate; "
            "persisted UNKNOWN source results use the separate source-evidence schema"
        )
    if not _is_digest(analysis["candidate_digest"]):
        raise ReplayError(f"{label}.candidate_digest is malformed")
    if analysis["control_digest"] is not None and not _is_digest(analysis["control_digest"]):
        raise ReplayError(f"{label}.control_digest is malformed")
    if analysis["borrowed_writes"] and analysis["outcome"] != "VIOLATION":
        raise ReplayError(f"{label} borrowed-write outcome is inconsistent")
    if not analysis["borrowed_writes"] and analysis["outcome"] != "SAFE_IN_FRAGMENT":
        raise ReplayError(f"{label} fresh-destination outcome is inconsistent")


def _integer(raw: Any, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ReplayError(f"{label} must be an integer")
    if minimum is not None and raw < minimum:
        raise ReplayError(f"{label} is below its minimum {minimum}")
    if maximum is not None and raw > maximum:
        raise ReplayError(f"{label} exceeds its maximum {maximum}")
    return raw


def _number(raw: Any, label: str, *, positive: bool = False) -> float:
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ReplayError(f"{label} must be a finite number")
    try:
        value = float(raw)
    except OverflowError as exc:
        raise ReplayError(f"{label} must be a finite number") from exc
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ReplayError(f"{label} is outside its nonnegative range")
    return value


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _bounded_structure(
    root: Any,
    *,
    signed: bool = False,
    maximum: int = MAX_STRUCTURE_ITEMS,
) -> None:
    """Walk only the JSON value types and reject cycles before descending.

    This is intentionally stricter than ``isinstance``.  A replay boundary must not
    invoke a mapping/list subclass supplied by a caller, and the length check must
    happen before copying or iterating a container.
    """

    stack: list[tuple[bool, Any, int]] = [(False, root, 0)]
    active: set[int] = set()
    pending = 1
    items = 0
    while stack:
        closing, value, depth = stack.pop()
        if closing:
            active.remove(value)
            continue
        pending -= 1
        items += 1
        if items > maximum:
            raise ReplayError("certificate structure exceeds the item limit")
        if depth > MAX_STRUCTURE_DEPTH:
            raise ReplayError("certificate structure exceeds the nesting limit")
        value_type = type(value)
        if value_type is str:
            try:
                encoded = value.encode("utf-8", "strict")
            except UnicodeEncodeError as exc:
                raise ReplayError("certificate contains invalid UTF-8") from exc
            if len(encoded) > MAX_STRING_BYTES:
                raise ReplayError("certificate string exceeds the byte limit")
            continue
        if value_type is int:
            if signed and not -(1 << 63) <= value < 1 << 63:
                raise ReplayError("certificate integer is outside signed-64")
            continue
        if value_type is float:
            if signed and not math.isfinite(value):
                raise ReplayError("certificate contains a non-finite number")
            continue
        if value is None or value_type is bool:
            continue
        if value_type not in {dict, list}:
            raise ReplayError(
                "certificate contains a non-JSON scalar or container "
                f"{value_type.__module__}.{value_type.__qualname__}"
            )
        identity = id(value)
        if identity in active:
            raise ReplayError("certificate structure contains an active-path cycle")
        size = len(value)
        contribution = 2 * size if value_type is dict else size
        if contribution > maximum or items + pending + contribution > maximum:
            raise ReplayError("certificate structure exceeds the item limit")
        active.add(identity)
        stack.append((True, identity, depth))
        if value_type is dict:
            keys = list(value)
            if len(keys) != size:
                raise ReplayError("certificate mapping changed during bounded walk")
            for key in reversed(keys):
                if type(key) is not str:
                    raise ReplayError("certificate object key is not an exact string")
                stack.append((False, value[key], depth + 1))
                stack.append((False, key, depth + 1))
        else:
            for item in reversed(value):
                stack.append((False, item, depth + 1))
        pending += contribution

def _closed(value, keys, label):
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ReplayError(f"{label} fields changed: {actual}")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()
