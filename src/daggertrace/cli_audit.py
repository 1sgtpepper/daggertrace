from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import cli_replay
from .canonical import sha256_file
from .cli_output import _audit_text, _emit
from .cli_replay import ReplayError, _discover_packet_paths
from .errors import DaggerTraceError

_MIN_WARMUPS = 1
_MAX_WARMUPS = 10
_MIN_SAMPLES = 3
_MAX_SAMPLES = 50
_ALL_TARGETS = "all"
# Keep parser bootstrap independent of the target router and producer modules.
_AUDIT_TARGETS = (
    "liger-rmsnorm",
    "openpipe-art-local-stats",
)
_SOURCE_TARGETS = frozenset({"triton-layernorm"})


def _audit(args: argparse.Namespace) -> int:
    _validate_benchmark_arguments(args.warmups, args.samples)
    targets = _selected_targets(args.target)
    output_root = _validated_output_path(args.output)
    target_root = _resolve_target_root(args.target_root)
    if target_root is not None and _paths_overlap(output_root, target_root):
        raise DaggerTraceError("audit output and target root must not overlap")
    if target_root is None:
        try:
            implicit_cache = (output_root.parent / ".target-cache").resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise DaggerTraceError(
                f"could not resolve the implicit target cache: {exc}"
            ) from exc
        if _paths_overlap(output_root, implicit_cache):
            raise DaggerTraceError("audit output and implicit target cache must not overlap")
    if len(targets) == 1:
        if _audit_output_needs_root_migration(output_root):
            summary = _audit_single_root(args, targets[0], output_root, target_root, None)
            _emit(summary, args.json, _audit_text(summary))
            return 0
        output = _validated_output_path(output_root / targets[0])
        summary = _audit_one(
            args,
            targets[0],
            output,
            target_root,
            None,
        )
        _emit(summary, args.json, _audit_text(summary))
        return 0

    summary = _audit_many(args, targets, output_root, target_root, None)
    _emit(summary, args.json, _audit_text(summary))
    return 0

def _audit_single_root(
    args: argparse.Namespace,
    target: str,
    output_root: Path,
    target_root: Path | None,
    semantic_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Migrate a legacy direct packet to the generated child-packet root layout."""

    root_staging = _fresh_staging_directory(output_root)
    packet_staging: Path | None = None
    completed = False
    try:
        packet_staging = _fresh_packet_staging(output_root, target)
        if semantic_validation is None:
            from .validation import run_semantic_validation

            semantic_validation = run_semantic_validation()
        summary = _build_audit_packet(
            args,
            target,
            packet_staging,
            output_root / target,
            target_root,
            semantic_validation,
        )
        _replace_path(packet_staging, root_staging / target)
        packet_staging = None
        _promote_directory(root_staging, output_root)
        completed = True
    except OSError as exc:
        raise DaggerTraceError(f"could not build the {target} audit artifact: {exc}") from exc
    finally:
        for staging in (packet_staging, root_staging):
            if staging is None:
                continue
            try:
                _discard_staging_directory(staging)
            except DaggerTraceError:
                if completed:
                    raise
    return summary


def _audit_one(
    args: argparse.Namespace,
    target: str,
    output: Path,
    target_root: Path | None,
    semantic_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    staging = _fresh_packet_staging(output.parent, target)
    completed = False
    try:
        if semantic_validation is None:
            from .validation import run_semantic_validation

            semantic_validation = run_semantic_validation()
        summary = _build_audit_packet(
            args,
            target,
            staging,
            output,
            target_root,
            semantic_validation,
        )
        _promote_directory(staging, output)
        completed = True
    except OSError as exc:
        raise DaggerTraceError(f"could not build the {target} audit artifact: {exc}") from exc
    finally:
        try:
            _discard_staging_directory(staging)
        except DaggerTraceError:
            if completed:
                raise
    return summary


def _audit_many(
    args: argparse.Namespace,
    targets: tuple[str, ...],
    output_root: Path,
    target_root: Path | None,
    semantic_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    if target_root is not None:
        for target in targets:
            _target_root_for(target_root, target, multiple=True)
    staging = _fresh_staging_directory(output_root)
    packet_stagings: list[Path] = []
    summaries: dict[str, dict[str, Any]] = {}
    completed = False
    try:
        if semantic_validation is None:
            from .validation import run_semantic_validation

            semantic_validation = run_semantic_validation()
        for target in targets:
            packet_staging = _fresh_packet_staging(output_root, target)
            packet_stagings.append(packet_staging)
            packet_output = output_root / target
            summaries[target] = _build_audit_packet(
                args,
                target,
                packet_staging,
                packet_output,
                _target_root_for(target_root, target, multiple=True),
                semantic_validation,
            )
            _replace_path(packet_staging, staging / target)
            packet_stagings.pop()
        _promote_directory(staging, output_root)
        completed = True
    except OSError as exc:
        raise DaggerTraceError(f"could not build the multi-target audit artifact: {exc}") from exc
    finally:
        try:
            _discard_staging_directory(staging)
        except DaggerTraceError:
            if completed:
                raise
        for packet_staging in packet_stagings:
            try:
                _discard_staging_directory(packet_staging)
            except DaggerTraceError:
                if completed:
                    raise
    return {
        "status": "VERIFIED MULTI-TARGET AUDIT",
        "directory": output_root.as_posix(),
        "target_count": len(summaries),
        "targets": summaries,
    }


def _build_audit_packet(
    args: argparse.Namespace,
    target: str,
    staging: Path,
    output: Path,
    target_root: Path | None,
    semantic_validation: dict[str, Any],
) -> dict[str, Any]:
    if target not in _AUDIT_TARGETS:
        raise AssertionError(f"unreachable certificate target {target!r}")

    from . import targets

    run = targets._audit_target(
        target,
        staging,
        target_root=target_root,
        warmups=args.warmups,
        samples=args.samples,
    )
    certificate_path = staging / "certificate.json"
    if run.get("control") is None:
        try:
            certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DaggerTraceError(
                f"{target} candidate-only adapter did not produce a readable certificate"
            ) from exc
        if not isinstance(certificate, dict):
            raise DaggerTraceError(
                f"{target} candidate-only adapter produced a non-object certificate"
            )
        if "control" not in certificate or certificate["control"] is not None:
            raise DaggerTraceError(
                f"{target} candidate-only certificate must contain a null control"
            )
        validation = certificate.get("validation")
        if (
            not isinstance(validation, dict)
            or validation.get("mutation_cases") != 0
            or validation.get("mutations_killed") != 0
        ):
            raise DaggerTraceError(
                f"{target} candidate-only certificate must record a zero paired mutation suite"
            )
        replay = cli_replay._replay_checked(certificate_path)
    else:
        from .certificate import make_certificate, write_certificate
        from .validation_mutations import validate_certificate_mutations

        validation = {
            **semantic_validation,
            "mutations_killed": semantic_validation["mutation_cases"],
        }
        certificate = make_certificate(
            candidate=run["candidate"],
            control=run["control"],
            analysis=run["analysis"],
            target=run["target"],
            runtime=run["runtime"],
            benchmark=run["benchmark"],
            controls=run["controls"],
            validation=validation,
        )
        write_certificate(
            certificate_path,
            certificate,
            reproducer_bytes=(staging / "reproducer.py").read_bytes(),
        )
        replay = cli_replay._replay_checked(certificate_path)
        killed = validate_certificate_mutations(
            certificate,
            certificate_path,
            replay_fn=cli_replay._replay_mutation,
        )
        if killed != semantic_validation["mutation_cases"]:
            raise AssertionError(f"certificate mutation total changed for {target}: {killed}")
        # Mutation tests use separate copies, leaving the validated packet unchanged.
    analysis = _analysis_record(run["analysis"])
    return {
        "status": _audit_status(analysis),
        "checker": certificate["checker"],
        "target_id": run.get("target_id", target),
        "target": run["target"],
        "analysis": analysis,
        "runtime": run["runtime"],
        "target_runtime": run.get("target_runtime"),
        "controls": run["controls"],
        "benchmark": run["benchmark"],
        "validation": validation,
        "replay": replay,
        "artifacts": {
            "directory": output.as_posix(),
            "certificate": (output / "certificate.json").as_posix(),
            "certificate_sha256": sha256_file(certificate_path),
            "candidate_capture": (output / "candidate.capture.json").as_posix(),
            "control_capture": (output / "control.capture.json").as_posix()
            if run.get("control") is not None
            else None,
            "reproducer": (output / "reproducer.py").as_posix(),
        },
    }


def _inspect_source_target(target: str, target_root: Path) -> dict[str, Any]:
    if target not in _SOURCE_TARGETS:
        raise DaggerTraceError(f"target {target!r} has no source classification")
    from .memory_model_target import audit_triton_layer_norm, _validate_source_result

    result = audit_triton_layer_norm(target_root)
    return _validate_source_result(result)


def _selected_targets(requested: str) -> tuple[str, ...]:
    if requested == _ALL_TARGETS:
        return _AUDIT_TARGETS
    if requested not in _AUDIT_TARGETS:
        raise DaggerTraceError(f"unsupported target {requested!r}")
    return (requested,)


def _target_root_for(
    target_root: Path | None,
    target: str,
    *,
    multiple: bool,
) -> Path | None:
    if target_root is None or not multiple:
        return target_root
    candidate = target_root / target
    try:
        if candidate.is_dir():
            return candidate
    except OSError as exc:
        raise DaggerTraceError(f"could not inspect the {target} target root: {exc}") from exc
    raise DaggerTraceError(
        f"multi-target --target-root must contain a pinned checkout at {candidate}"
    )


def _analysis_record(analysis: Any) -> dict[str, Any]:
    if hasattr(analysis, "to_dict"):
        analysis = analysis.to_dict()
    if not isinstance(analysis, dict):
        raise AssertionError("target analysis must be a mapping or expose to_dict()")
    return analysis


def _audit_status(analysis: dict[str, Any]) -> str:
    outcome = analysis.get("outcome", "UNKNOWN")
    if analysis.get("collapse") is not None:
        return f"VERIFIED {outcome} + COLLAPSE"
    return f"VERIFIED {outcome}"


def _validated_output_path(requested: Path) -> Path:
    try:
        expanded = requested.expanduser()
        output = expanded.resolve(strict=False)
        protected = (Path.cwd().resolve(), Path.home().resolve())
        unresolved = expanded if expanded.is_absolute() else Path.cwd() / expanded
    except (OSError, RuntimeError) as exc:
        raise DaggerTraceError(f"could not resolve the audit output path: {exc}") from exc
    if (
        output.parent == output
        or output.parent.parent == output.parent
        or any(path.is_relative_to(output) for path in protected)
    ):
        raise DaggerTraceError(f"refusing to replace unsafe audit output {output}")
    if _contains_symlink(unresolved):
        raise DaggerTraceError(f"refusing to replace symlinked audit output {requested}")
    return output


def _resolve_target_root(requested: Path | None) -> Path | None:
    if requested is None:
        return None
    try:
        return requested.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise DaggerTraceError(f"could not resolve the audit target path: {exc}") from exc


def _contains_symlink(path: Path) -> bool:
    current = path
    while True:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _paths_overlap(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def _validate_benchmark_arguments(warmups: int, samples: int) -> None:
    if type(warmups) is not int or not _MIN_WARMUPS <= warmups <= _MAX_WARMUPS:
        raise DaggerTraceError("audit warmups must be an integer from 1 through 10")
    if type(samples) is not int or not _MIN_SAMPLES <= samples <= _MAX_SAMPLES:
        raise DaggerTraceError("audit samples must be an integer from 3 through 50")


def _audit_output_needs_root_migration(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        packets = _discover_packet_paths(path, depth=0)
    except ReplayError as exc:
        raise DaggerTraceError(
            f"existing audit output is not a valid packet root: {exc}"
        ) from exc
    return bool(packets and packets[0].parent == path)

def _fresh_staging_directory(output: Path) -> Path:
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        return Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.staging-",
                dir=output.parent,
            )
        )
    except OSError as exc:
        raise DaggerTraceError(f"could not create an audit staging directory: {exc}") from exc


def _fresh_packet_staging(output_root: Path, target: str) -> Path:
    """Stage one packet beside the final root so the shared target cache survives."""

    prefix = f".{output_root.name}.{target}.staging-"
    try:
        output_root.parent.mkdir(parents=True, exist_ok=True)
        for sibling in output_root.parent.iterdir():
            if sibling.is_dir() and sibling.name.startswith(prefix):
                raise DaggerTraceError(
                    f"could not create a {target} packet staging directory; "
                    f"recoverable transaction residue: {sibling}"
                )
        return Path(
            tempfile.mkdtemp(
                prefix=prefix,
                dir=output_root.parent,
            )
        )
    except OSError as exc:
        raise DaggerTraceError(f"could not create a {target} packet staging directory: {exc}") from exc


def _promote_directory(staging: Path, output: Path) -> None:
    backup: Path | None = None
    try:
        staging = staging.absolute()
        output = output.absolute()
        output.parent.mkdir(parents=True, exist_ok=True)
        for sibling in output.parent.iterdir():
            if sibling == staging or not sibling.is_dir():
                continue
            if sibling.name.startswith(
                (f".{output.name}.backup-", f".{output.name}.staging-")
            ):
                raise DaggerTraceError(
                    "could not replace the audit output; recoverable transaction "
                    f"residue: {sibling}"
                )
        if output.is_dir() and not output.is_symlink():
            for child in output.iterdir():
                if child.is_dir() and (
                    child.match(".*.backup-*") or child.match(".*.staging-*")
                ):
                    raise DaggerTraceError(
                        "could not replace the audit output; recoverable transaction "
                        f"residue: {child}"
                    )
        if output.exists() or output.is_symlink():
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{output.name}.backup-",
                    dir=output.parent,
                )
            ) / "previous"
    except OSError as exc:
        raise DaggerTraceError(f"could not replace the audit output: {exc}") from exc

    if backup is not None:
        try:
            _replace_path(output, backup)
        except OSError as promotion_error:
            if backup.parent.exists():
                try:
                    backup.parent.rmdir()
                except OSError as cleanup_error:
                    raise DaggerTraceError(
                        "could not replace the audit output: "
                        f"{promotion_error}; cleanup failed; recoverable residue: "
                        f"{backup.parent}: {cleanup_error}"
                    ) from promotion_error
            raise DaggerTraceError(
                f"could not replace the audit output: {promotion_error}"
            ) from promotion_error

    try:
        _replace_path(staging, output)
    except OSError as promotion_error:
        if backup is not None:
            try:
                _replace_path(backup, output)
            except OSError as rollback_error:
                raise DaggerTraceError(
                    "could not promote the audit artifact and could not restore "
                    f"the prior output from {backup}: {rollback_error}"
                ) from promotion_error
            if backup.parent.exists():
                try:
                    backup.parent.rmdir()
                except OSError as cleanup_error:
                    raise DaggerTraceError(
                        "could not atomically promote the audit artifact: "
                        f"{promotion_error}; cleanup failed; recoverable residue: "
                        f"{backup.parent}: {cleanup_error}"
                    ) from promotion_error
        raise DaggerTraceError(
            f"could not atomically promote the audit artifact: {promotion_error}"
        ) from promotion_error

    if backup is not None:
        try:
            _delete_exact_path(backup)
            if backup.parent.exists():
                backup.parent.rmdir()
        except OSError as cleanup_error:
            raise DaggerTraceError(
                "audit artifact promotion succeeded; cleanup failed; "
                f"recoverable residue: {backup.parent}: {cleanup_error}"
            ) from cleanup_error


def _replace_path(source: Path, destination: Path) -> None:
    source.replace(destination)


def _delete_exact_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _discard_staging_directory(staging: Path) -> None:
    try:
        if staging.exists():
            shutil.rmtree(staging)
    except OSError as exc:
        raise DaggerTraceError(f"could not remove audit staging directory {staging}: {exc}") from exc
