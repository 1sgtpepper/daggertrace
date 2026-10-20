"""V2 ownership-analysis entry point."""

from __future__ import annotations

from typing import Any

from .errors import InvalidArtifact
from .model import Analysis, Capture
from .ownership_v2 import _analyze_v2
from .version import CAPTURE_SCHEMA


def analyze(
    candidate: Capture | dict[str, Any],
    control: Capture | dict[str, Any] | None = None,
) -> Analysis:
    """Analyze one live/default v2 capture, optionally against its paired control."""

    if type(candidate) is dict:
        candidate = Capture.from_dict(candidate)
    if type(control) is dict:
        control = Capture.from_dict(control)
    if type(candidate) is not Capture or (
        control is not None and type(control) is not Capture
    ):
        raise InvalidArtifact("capture roots must be exact Capture objects or dictionaries")
    values = (candidate,) if control is None else (candidate, control)
    schemas = tuple(object.__getattribute__(value, "schema") for value in values)
    if any(type(schema) is not str or schema != CAPTURE_SCHEMA for schema in schemas):
        raise InvalidArtifact("capture root schema is unsupported")
    return _analyze_v2(candidate, control)
