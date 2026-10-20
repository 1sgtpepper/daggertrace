"""Load target adapters on demand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import TargetError


_TARGET_NAMES = (
    "liger-rmsnorm",
    "openpipe-art-local-stats",
)

def run_liger_reproducer(target_root: Path | None = None) -> dict[str, Any]:
    from .liger_target import run_liger_reproducer as reproduce

    return reproduce(target_root)


def _audit_target(
    name: str,
    output_dir: Path,
    *,
    target_root: Path | None = None,
    warmups: int = 3,
    samples: int = 10,
) -> dict[str, Any]:
    if name == "liger-rmsnorm":
        from .liger_target import audit_liger

        return audit_liger(
            output_dir,
            target_root=target_root,
            warmups=warmups,
            samples=samples,
        )
    if name == "openpipe-art-local-stats":
        from .art_target import audit_art

        return audit_art(
            output_dir,
            target_root=target_root,
            warmups=warmups,
            samples=samples,
        )
    raise TargetError(f"unknown target {name!r}; choose one of {_TARGET_NAMES}")


def _run_target_reproducer(name: str, target_root: Path | None = None) -> dict[str, Any]:
    if name == "liger-rmsnorm":
        return run_liger_reproducer(target_root)
    if name == "openpipe-art-local-stats":
        from .art_target import run_art_reproducer

        return run_art_reproducer(target_root)
    raise TargetError(f"unknown target {name!r}; choose one of {_TARGET_NAMES}")


def _replay_target(
    certificate: dict[str, Any],
    certificate_path: Path,
    generic_result: dict[str, Any],
) -> dict[str, Any]:
    """Apply target-specific checks after generic replay succeeds."""

    target = certificate.get("target")
    if not isinstance(target, dict):
        raise TargetError("certificate target identity is missing")
    repository = target.get("repository")
    if repository == "https://github.com/OpenPipe/ART":
        from .art_target import _replay_art_loaded

        return _replay_art_loaded(certificate, certificate_path, generic_result)

    from .liger_packet import LIGER_REPOSITORY

    if repository == LIGER_REPOSITORY:
        from .liger_target import _replay_liger_loaded

        return _replay_liger_loaded(
            certificate,
            certificate_path,
            generic_result,
        )
    raise TargetError(f"unsupported certificate target repository {repository!r}")
