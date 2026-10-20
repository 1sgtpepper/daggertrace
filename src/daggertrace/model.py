# ruff: noqa: F401

"""Stable v2 artifact-model imports."""

# Re-exported dataclasses resolve deferred annotations in this module.
from typing import Any

from .model_core import (
    Analysis,
    BorrowedWrite,
    Capture,
    Collapse,
    Graph,
    Kernel,
    Node,
    Outcome,
    Reason,
    TensorMeta,
    ref,
    walk_refs,
)


for _model_type in (
    Outcome,
    Reason,
    TensorMeta,
    Node,
    Graph,
    Kernel,
    Capture,
    BorrowedWrite,
    Collapse,
    Analysis,
):
    _model_type.__module__ = __name__

del _model_type
