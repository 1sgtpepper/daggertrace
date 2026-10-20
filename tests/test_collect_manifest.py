from __future__ import annotations

import json
import sys

import pytest

import evidence.collect_manifest as manifest


def test_manifest_cli_rejects_symlinked_output_without_writing_outside_bundle(
    monkeypatch, tmp_path
):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"keep these bytes\n")
    output = root / "manifest.json"
    output.symlink_to(outside)
    monkeypatch.setattr(
        sys,
        "argv",
        ["collect_manifest", "--root", str(root), "--output", str(output)],
    )

    with pytest.raises(RuntimeError, match="output must not be a symlink"):
        manifest.main()

    assert outside.read_bytes() == b"keep these bytes\n"


def test_manifest_cli_excludes_regular_output_from_manifest(monkeypatch, tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    output = root / "manifest.json"
    output.write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["collect_manifest", "--root", str(root), "--output", str(output)],
    )

    assert manifest.main() == 0

    result = json.loads(output.read_text(encoding="utf-8"))
    assert [entry["path"] for entry in result["files"]] == ["payload.txt"]


def test_manifest_rejects_symlink_member_even_when_it_resolves_to_excluded_output(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    output = root / "manifest.json"
    output.write_text("stale\n", encoding="utf-8")
    (root / "alias").symlink_to(output)

    with pytest.raises(RuntimeError, match="bundle contains a symlink"):
        manifest.collect(root, output)


@pytest.mark.parametrize("root_kind", ("missing", "file", "symlink"))
def test_manifest_rejects_invalid_bundle_roots(tmp_path, root_kind):
    root = tmp_path / "bundle"
    if root_kind == "file":
        root.write_text("not a bundle\n", encoding="utf-8")
    elif root_kind == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="root must be a regular directory"):
        manifest.collect(root)
