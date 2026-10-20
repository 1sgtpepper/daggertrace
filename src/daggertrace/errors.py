from __future__ import annotations


class DaggerTraceError(Exception):
    """Base class for deterministic DaggerTrace failures."""


class InvalidArtifact(DaggerTraceError):
    """A persisted graph or certificate is malformed or hash-invalid."""


class UnsupportedCapture(DaggerTraceError):
    """The captured program is outside the closed ownership profile."""


class TargetError(DaggerTraceError):
    """A pinned external target could not be obtained or matched exactly."""
