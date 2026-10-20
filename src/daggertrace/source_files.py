"""Neutral verified reads for pinned source trees."""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
import types
from pathlib import Path, PurePosixPath, PureWindowsPath

from .canonical import sha256_bytes
from .errors import TargetError


MAX_PINNED_FILE_BYTES = 8 << 20


def _load_exact_module(
    name: str,
    path: Path,
    expected_sha256: str,
    target_name: str = "target",
):
    source = _read_regular_file(path, path, target_name)
    actual = sha256_bytes(source)
    if actual != expected_sha256:
        raise TargetError(f"pinned {target_name} source hash mismatch for {path}: {actual}")
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    if spec is None:
        raise TargetError(f"cannot create an import spec for {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__spec__ = spec
    module.__loader__ = None
    module.__cached__ = None
    sys.modules[name] = module
    try:
        code = compile(source, str(path), "exec")
        exec(code, module.__dict__)
        if Path(module.__file__).resolve() != path.resolve():
            raise TargetError(f"loaded {name} from unexpected path {module.__file__}")
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _read_verified_tree_file(
    root: Path,
    relative: str,
    expected_sha256: str,
    target_name: str = "target",
) -> bytes:
    path = _validated_tree_file_path(root, relative, target_name)
    data = _read_regular_file(path, relative, target_name)
    actual = sha256_bytes(data)
    if actual != expected_sha256:
        raise TargetError(f"pinned {target_name} source hash mismatch for {relative}: {actual}")
    return data


def _validated_tree_file_path(root: Path, relative: str, target_name: str) -> Path:
    parts = _tree_path_parts(relative, target_name)
    try:
        root_metadata = os.lstat(_filesystem_path(root))
    except OSError as exc:
        raise TargetError(f"pinned {target_name} tree is absent: {root}") from exc
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise TargetError(f"pinned {target_name} tree is not a directory: {root}")

    path = root
    for index, part in enumerate(parts):
        path /= part
        try:
            metadata = os.lstat(_filesystem_path(path))
        except OSError as exc:
            raise TargetError(f"pinned {target_name} tree lacks {relative}") from exc
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise TargetError(
                f"pinned {target_name} tree path component is not a directory: {relative}"
            )
        if index == len(parts) - 1 and not stat.S_ISREG(metadata.st_mode):
            raise TargetError(f"pinned {target_name} source is not a regular file: {relative}")
    return path


def _tree_path_parts(relative: str, target_name: str) -> tuple[str, ...]:
    invalid = f"invalid pinned {target_name} tree path: {relative}"
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise TargetError(invalid)
    try:
        relative.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TargetError(invalid) from exc
    path = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    parts = path.parts
    if (
        path.is_absolute()
        or path.as_posix() != relative
        or windows.drive
        or windows.root
        or not parts
        or any(part in {"", ".", ".."} or PureWindowsPath(part).drive for part in parts)
    ):
        raise TargetError(invalid)
    return parts


def _read_regular_file(path: Path, label: object, target_name: str = "target") -> bytes:
    try:
        metadata = os.lstat(_filesystem_path(path))
    except OSError as exc:
        raise TargetError(f"pinned {target_name} file is absent: {label}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise TargetError(f"pinned {target_name} file is not a regular file: {label}")

    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(_filesystem_path(path), flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise TargetError(f"pinned {target_name} file is not a regular file: {label}")
        if metadata.st_size > MAX_PINNED_FILE_BYTES:
            raise TargetError(
                f"pinned {target_name} file exceeds {MAX_PINNED_FILE_BYTES} bytes: {label}"
            )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            data = handle.read(MAX_PINNED_FILE_BYTES + 1)
            if len(data) > MAX_PINNED_FILE_BYTES:
                raise TargetError(
                    f"pinned {target_name} file exceeds {MAX_PINNED_FILE_BYTES} bytes: {label}"
                )
            return data
    except TargetError:
        raise
    except OSError as exc:
        raise TargetError(f"could not read pinned {target_name} file {label}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _filesystem_path(path: Path) -> str | bytes:
    value = os.fspath(path)
    try:
        os.fsencode(value)
    except UnicodeEncodeError:
        if os.name == "posix":
            return value.encode("utf-8", "surrogateescape")
        raise
    return value
