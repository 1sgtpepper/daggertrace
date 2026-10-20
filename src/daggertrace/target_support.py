"""Shared archive, environment, and runtime helpers for target adapters.

Target-specific validation stays in the target adapters.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import platform
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .capture import _require_framework
from .errors import TargetError, UnsupportedCapture

_MAX_ARCHIVE_BYTES = 64 << 20
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_ARCHIVE_EXPANDED_BYTES = 512 << 20


def _counter_snapshot(torch: Any) -> dict[str, dict[str, int]]:
    return {
        namespace: dict(torch._dynamo.utils.counters[namespace])
        for namespace in ("aot_autograd", "inductor")
    }


def _counter_delta(
    before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    return {
        namespace: {
            key: after[namespace].get(key, 0) - before[namespace].get(key, 0)
            for key in sorted(set(before[namespace]) | set(after[namespace]))
            if after[namespace].get(key, 0) != before[namespace].get(key, 0)
        }
        for namespace in sorted(before)
    }


def _max_abs(left: Any, right: Any) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _tensor_hash(tensor: Any) -> str:
    data = (
        tensor.detach()
        .contiguous()
        .view(dtype=__import__("torch").uint8)
        .flatten()
        .cpu()
        .tolist()
    )
    return hashlib.sha256(bytes(data)).hexdigest()


def _require_exact_gpu_framework(torch: Any, triton: Any, target: str) -> None:
    try:
        _require_framework(
            torch.__version__,
            torch.version.git_version,
            triton.__version__,
        )
    except (AttributeError, UnsupportedCapture) as exc:
        raise TargetError(f"{target} requires the exact supported GPU framework: {exc}") from exc


def _require_generic_target_result(
    result: dict[str, Any],
    *,
    outcome: str,
    reasons: list[str],
    borrowed_writes: int,
    contextual_collapse: bool,
    target: str,
) -> None:
    expected = {
        "valid": True,
        "outcome": outcome,
        "reasons": reasons,
        "borrowed_writes": borrowed_writes,
        "contextual_collapse": contextual_collapse,
    }
    if not isinstance(result, dict) or any(result.get(key) != value for key, value in expected.items()):
        raise TargetError(f"{target} generic replay result does not match its target contract")


def _fetch_pinned_archive(
    cache_dir: Path,
    *,
    root_name: str,
    repository: str,
    revision: str,
    user_agent: str,
    label: str,
    validate_tree,
) -> Path:
    root = cache_dir / root_name
    if root.is_symlink():
        raise TargetError(f"refusing symlinked {label} cache entry {root}")
    if root.is_dir():
        validate_tree(root)
        return root
    if root.exists():
        raise TargetError(f"{label} cache entry is not a directory: {root}")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TargetError(f"could not prepare the {label} target cache: {exc}") from exc
    url = f"{repository}/archive/{revision}.zip"
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError as exc:
                    raise TargetError(
                        f"pinned {label} archive has an invalid Content-Length"
                    ) from exc
                if declared_size < 0:
                    raise TargetError(f"pinned {label} archive has an invalid Content-Length")
                if declared_size > _MAX_ARCHIVE_BYTES:
                    raise TargetError(f"pinned {label} archive exceeds the download limit")
            payload = response.read(_MAX_ARCHIVE_BYTES + 1)
            if len(payload) > _MAX_ARCHIVE_BYTES:
                raise TargetError(f"pinned {label} archive exceeds the download limit")
    except OSError as exc:
        raise TargetError(f"could not fetch pinned {label} archive: {exc}") from exc
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{root.name}.staging-",
                dir=cache_dir,
            )
        )
    except OSError as exc:
        raise TargetError(f"could not create a {label} cache staging directory: {exc}") from exc
    completed = False
    try:
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                members = archive.infolist()
                if not members:
                    raise TargetError(f"pinned {label} archive is empty")
                if len(members) > _MAX_ARCHIVE_MEMBERS:
                    raise TargetError(f"pinned {label} archive has too many members")
                if sum(member.file_size for member in members) > _MAX_ARCHIVE_EXPANDED_BYTES:
                    raise TargetError(f"pinned {label} archive exceeds the expansion limit")
                expected_prefix = f"{root_name}/"
                staging_root = staging.resolve()
                for member in members:
                    if not member.filename.startswith(expected_prefix):
                        raise TargetError(f"unexpected {label} archive member {member.filename!r}")
                    destination = (staging / member.filename).resolve()
                    if not destination.is_relative_to(staging_root):
                        raise TargetError(f"unsafe {label} archive member {member.filename!r}")
                archive.extractall(staging)
        except zipfile.BadZipFile as exc:
            raise TargetError(f"pinned {label} download is not a valid ZIP archive") from exc
        except OSError as exc:
            raise TargetError(f"could not extract pinned {label} archive: {exc}") from exc

        staged_root = staging / root.name
        validate_tree(staged_root)
        if root.is_symlink():
            raise TargetError(f"refusing symlinked {label} cache entry {root}")
        if root.is_dir():
            validate_tree(root)
            completed = True
            return root
        if root.exists():
            raise TargetError(f"{label} cache entry is not a directory: {root}")
        try:
            staged_root.replace(root)
        except OSError as exc:
            raise TargetError(
                f"could not atomically populate the {label} target cache: {exc}"
            ) from exc
        completed = True
        return root
    finally:
        try:
            if staging.exists():
                shutil.rmtree(staging)
        except OSError as exc:
            if completed:
                raise TargetError(
                    f"could not remove {label} cache staging directory {staging}: {exc}"
                ) from exc

def _validate_benchmark_bounds(warmups: int, samples: int, target_name: str = "Liger") -> None:
    if type(warmups) is not int or not 1 <= warmups <= 10:
        raise TargetError(f"{target_name} benchmark warmups must be an integer from 1 through 10")
    if type(samples) is not int or not 3 <= samples <= 50:
        raise TargetError(f"{target_name} benchmark samples must be an integer from 3 through 50")


def _gpu_environment(target_name: str):
    torch = _require_module("torch", "PyTorch", target_name)
    triton = _require_module("triton", "Triton", target_name)

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": properties.name,
        "capability": list(torch.cuda.get_device_capability(device_index)),
        "total_memory_bytes": properties.total_memory,
        "multiprocessors": properties.multi_processor_count,
        "driver": _nvidia_driver(),
    }


def _require_module(name: str, display_name: str, target_name: str = "Liger"):
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as exc:
        raise TargetError(f"the {target_name} demonstration requires {display_name}") from exc


def _nvidia_driver() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.splitlines()[0].strip()
