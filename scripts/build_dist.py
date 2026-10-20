#!/usr/bin/env python3
"""Build distributions with exact source-revision provenance."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVISION_FILE = ROOT / "SOURCE_REVISION"


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    generated_revision = False
    if (ROOT / ".git").exists():
        if _git("status", "--porcelain"):
            raise SystemExit("distribution builds require a clean worktree")
        revision = _git("rev-parse", "HEAD")
        if REVISION_FILE.exists():
            raise SystemExit("remove the stale SOURCE_REVISION file before building")
        REVISION_FILE.write_text(revision + "\n", encoding="utf-8")
        generated_revision = True
    else:
        try:
            revision = REVISION_FILE.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(
                "distribution builds require Git metadata or a SOURCE_REVISION file"
            ) from exc

    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise SystemExit("SOURCE_REVISION must contain a lowercase 40-hex revision")

    try:
        return subprocess.run(
            [sys.executable, "-m", "build", *sys.argv[1:]],
            cwd=ROOT,
            check=False,
        ).returncode
    finally:
        if generated_revision:
            REVISION_FILE.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
