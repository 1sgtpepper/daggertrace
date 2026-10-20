from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = (
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    ROOT / "artifacts" / "README.md",
    ROOT / "evidence" / "README.md",
    ROOT / "evidence" / "results" / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md",
    *sorted((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.md")),
    *sorted((ROOT / "docs").glob("*.md")),
)


def test_documented_source_result_has_the_complete_wire_shape():
    text = (ROOT / "docs" / "CERTIFICATES.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```json\s+(.*?)\s+```", text, re.DOTALL)
    assert [json.loads(block) for block in blocks] == [{
        "outcome": "UNKNOWN",
        "reasons": ["MEMORY_MODEL"],
        "borrowed_writes": [],
        "collapse": None,
        "unknown_detail": (
            "UNKNOWN:MEMORY_MODEL: source contains a Triton memory protocol operation"
        ),
    }]


def test_documented_local_markdown_links_resolve():
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for document in PUBLIC_FILES:
        for raw in pattern.findall(document.read_text(encoding="utf-8")):
            target = raw.split("#", 1)[0]
            if not target or "://" in target:
                continue
            assert (document.parent / target).resolve().exists(), (document, raw)


def test_public_docs_contain_no_machine_local_paths():
    text = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_FILES)
    assert "/" + "home/" not in text
    assert "file" + "://" not in text
