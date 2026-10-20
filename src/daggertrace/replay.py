"""Independent literal replay facade for ownership certificates.

The implementation is split by packet boundary, source profile, and wire version.
Every dependency remains standard-library-only plus the local protocol vocabulary.
"""

from .replay_packet import (  # noqa: F401
    CAPTURE_SCHEMA,
    CONFIG_COUNT_MAX,
    EXPECTED_CAPTURE_CONFIG,
    MAX_CERTIFICATE_BYTES,
    MAX_DERIVED_PARTS,
    MAX_EXTERNAL_FILES,
    MAX_EXTERNAL_FILE_BYTES,
    MAX_KERNELS,
    MAX_PACKET_BYTES,
    MAX_PATH_BYTES,
    MAX_PATH_COMPONENTS,
    MAX_SOURCE_BYTES,
    MAX_STRING_BYTES,
    MAX_STRUCTURE_DEPTH,
    MAX_STRUCTURE_ITEMS,
    PROFILE,
    ReplayError,
    SCHEMA,
    SOURCE_EVIDENCE_SCHEMA,
    SUPPORTED_TORCH,
    SUPPORTED_TORCH_GIT,
    SUPPORTED_TRITON,
)
from .replay_core import (  # noqa: F401
    replay_certificate,
    replay_source_evidence,
)


for _public_replay_object in (ReplayError, replay_certificate, replay_source_evidence):
    _public_replay_object.__module__ = __name__

del _public_replay_object
