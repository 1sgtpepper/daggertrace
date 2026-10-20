"""Target-neutral Triton protocol classifications shared by source paths."""

from __future__ import annotations

from typing import Final


TRITON_PROTOCOL_OPERATIONS: Final = {
    "atomic_add": "atomic_add",
    "atomic_and": "atomic_and",
    "atomic_cas": "atomic_cas",
    "atomic_max": "atomic_max",
    "atomic_min": "atomic_min",
    "atomic_or": "atomic_or",
    "atomic_xchg": "atomic_xchg",
    "atomic_xor": "atomic_xor",
    "barrier": "barrier",
    "debug_barrier": "barrier",
}

_TRITON_SOURCE_TL_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "arange",
        "exp",
        "exp2",
        "expand_dims",
        "join",
        "load",
        "log2",
        "max",
        "program_id",
        "split",
        "static_range",
        "store",
        "sum",
        "where",
        "zeros",
    }
)
_TRITON_SOURCE_BARE_CALLS: Final[frozenset[str]] = frozenset(
    {"float", "min", "range", "rsqrt"}
)

MEMORY_MODEL_REASON: Final = "MEMORY_MODEL"
MEMORY_MODEL_STATUS: Final = f"UNKNOWN:{MEMORY_MODEL_REASON}"
MEMORY_MODEL_PROTOCOL_DETAIL: Final = "source contains a Triton memory protocol operation"
UNKNOWN_MEMORY_MODEL_DETAIL: Final = f"{MEMORY_MODEL_STATUS}: {MEMORY_MODEL_PROTOCOL_DETAIL}"


def triton_protocol_kind(operation: str) -> str | None:
    """Return the canonical protocol kind for a Triton operation, if any."""

    if not operation.startswith("tl."):
        return None
    return TRITON_PROTOCOL_OPERATIONS.get(operation.removeprefix("tl."))
