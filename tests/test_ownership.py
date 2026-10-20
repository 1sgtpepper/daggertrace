from __future__ import annotations

from dataclasses import replace

import pytest

from daggertrace.errors import InvalidArtifact
from daggertrace.model import Outcome
from daggertrace.ownership import analyze
from daggertrace.validation_v2_fixtures import _micrograph

def test_analyze_dispatches_exact_raw_routes_and_rejects_unknown_typed_schema():
    candidate, _ = _micrograph(False, "direct", False, 1, False)
    assert analyze(candidate.to_dict()).outcome == Outcome.SAFE_IN_FRAGMENT
    with pytest.raises(InvalidArtifact):
        analyze(replace(candidate, schema="unknown"))
