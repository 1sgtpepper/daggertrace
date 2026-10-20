from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cli_audit import (
    _ALL_TARGETS,
    _AUDIT_TARGETS,
    _MAX_SAMPLES,
    _MAX_WARMUPS,
    _MIN_SAMPLES,
    _MIN_WARMUPS,
    _SOURCE_TARGETS,
    _audit,
    _inspect_source_target,
    _resolve_target_root,
    _selected_targets,
    _target_root_for,
)
from .cli_output import (
    _emit,
    _reproducer_text,
    _source_classification_text,
    _validation_text,
    _write_text,
)
from .cli_replay import (
    ReplayError,
    _packet_paths,
    _replay_packet,
    _replay_summary,
    replay_certificate,  # noqa: F401
)
from .errors import DaggerTraceError
from .version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daggertrace",
        description="Check for writes to incoming gradients in PyTorch GPU backward functions",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="capture and analyze one or all supported examples")
    audit.add_argument("target", choices=[*_AUDIT_TARGETS, _ALL_TARGETS])
    audit.add_argument(
        "--gpu",
        action="store_true",
        help="acknowledge that this command runs CUDA",
    )
    audit.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/final"),
        help=(
            "replace this packet or packet root; a single target is written "
            "under <output>/<target> (default: artifacts/final)"
        ),
    )
    audit.add_argument(
        "--target-root",
        type=Path,
        help=(
            "use an exact pinned external target; otherwise audit uses "
            "<output-parent>/.target-cache (external targets only)"
        ),
    )
    audit.add_argument(
        "--warmups",
        type=int,
        choices=range(_MIN_WARMUPS, _MAX_WARMUPS + 1),
        default=3,
        help="CUDA benchmark warmup iterations (1..10)",
    )
    audit.add_argument(
        "--samples",
        type=int,
        choices=range(_MIN_SAMPLES, _MAX_SAMPLES + 1),
        default=10,
        help="CUDA benchmark samples (3..50)",
    )
    audit.add_argument("--json", action="store_true", help="print the result as JSON")

    inspect = subparsers.add_parser(
        "inspect", help="classify source code without running a kernel or writing a packet"
    )
    inspect.add_argument("target", choices=sorted(_SOURCE_TARGETS))
    inspect.add_argument(
        "--target-root",
        type=Path,
        help="required exact pinned external target checkout for source classification",
    )
    inspect.add_argument("--json", action="store_true", help="print the result as JSON")

    replay = subparsers.add_parser(
        "replay", help="independently replay one certificate/source packet or a packet root"
    )
    replay.add_argument(
        "certificate",
        metavar="PACKET",
        type=Path,
        help="certificate, source-evidence file, packet directory, or packet root",
    )
    replay.add_argument("--json", action="store_true", help="print the replay result as JSON")

    validate = subparsers.add_parser("validate", help="run semantic tests without CUDA")
    validate.add_argument("--json", action="store_true")

    reproduce = subparsers.add_parser(
        "reproduce", help="run the minimal reproducer for a supported example"
    )
    reproduce.add_argument("target", choices=[*_AUDIT_TARGETS, _ALL_TARGETS])
    reproduce.add_argument("--gpu", action="store_true", help="acknowledge that this command runs CUDA")
    reproduce.add_argument(
        "--target-root",
        type=Path,
        help="required exact pinned checkout for external target reproducers",
    )
    reproduce.add_argument("--json", action="store_true", help="print the result as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            if not args.gpu:
                raise DaggerTraceError("certificate target selection requires explicit --gpu")
            return _audit(args)
        if args.command == "inspect":
            target_root = _resolve_target_root(args.target_root)
            if target_root is None:
                raise DaggerTraceError(
                    f"{args.target} requires --target-root for its pinned source checkout"
                )
            result = _inspect_source_target(args.target, target_root)
            _emit(result, args.json, _source_classification_text(args.target, result))
            return 0
        if args.command == "replay":
            paths = _packet_paths(args.certificate)
            results = [
                (path, _replay_packet(path))
                for path in paths
            ]
            first_path = paths[0]
            aggregate = first_path != args.certificate and first_path.parent != args.certificate
            value, human = _replay_summary(results, aggregate=aggregate)
            _emit(value, args.json, human)
            return 0
        if args.command == "validate":
            from .validation import run_semantic_validation

            result = run_semantic_validation()
            _emit(result, args.json, _validation_text(result))
            return 0
        if args.command == "reproduce":
            selected = _selected_targets(args.target)
            if not args.gpu:
                raise DaggerTraceError("certificate target reproducers require explicit --gpu")
            if args.target_root is None:
                raise DaggerTraceError(
                    "external target reproducers require an explicit --target-root "
                    "for the exact pinned checkout"
                )
            target_root = _resolve_target_root(args.target_root)
            target_roots = {
                target: _target_root_for(
                    target_root,
                    target,
                    multiple=args.target == _ALL_TARGETS,
                )
                for target in selected
            }
            from . import targets

            results = []
            for target in selected:
                result = targets._run_target_reproducer(target, target_roots[target])
                if not isinstance(result, dict):
                    raise DaggerTraceError(f"{target} reproducer returned a non-object result")
                result = {**result, "target_id": target}
                results.append((target, result))
            if len(results) == 1:
                value, human = results[0][1], _reproducer_text(results[0][1])
            else:
                value = {
                    "status": "VERIFIED",
                    "targets": {target: result for target, result in results},
                }
                sections = [
                    "STATUS VERIFIED",
                    f"TARGETS {len(results)}",
                ]
                sections.extend(
                    f"TARGET {target}\n{_reproducer_text(result)}"
                    for target, result in results
                )
                human = "\n\n".join(sections)
            _emit(value, args.json, human)
            return 0
    except KeyboardInterrupt:
        _write_text("daggertrace: interrupted", sys.stderr)
        return 130
    except ReplayError as exc:
        _write_text(f"daggertrace: replay rejected: {exc}", sys.stderr)
        return 1
    except DaggerTraceError as exc:
        _write_text(f"daggertrace: {exc}", sys.stderr)
        return 2
    raise AssertionError(f"unreachable command {args.command}")
