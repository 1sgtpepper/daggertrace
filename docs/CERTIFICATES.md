# Certificates and replay

A certificate binds an ownership judgment to exact supplied packet bytes. The
supported schema is `daggertrace.cotangent-ownership/2`. Digests establish byte
equality and internal consistency; they are one replay input, not `VALID` by
themselves, signatures, or proof of runtime provenance. The offline-v3
certificate format and its `seal`/`ingest` workflow are retired; v2 packet
writing still computes canonical digests.

## Certificate object

The top-level object has exactly:

```text
schema, checker, target, candidate, control, analysis,
runtime, benchmark, controls, validation, digest
```

`candidate` is required. `control` is the normalized capture paired with the
candidate, or `null`.
`analysis` records the outcome, reasons, borrowed writes, candidate/control
capture digests, and optional details about contribution collapse. Runtime and
benchmark fields are supporting records, not substitutes for semantic replay.
Candidate-only ART packets have empty additional `controls`.

`target` has exactly `repository`, `revision`, `files`, `kernels`, and
`specialization`. Each `target.files` entry declares
`packet_path, target_path, size, sha256, role`. Each `target.kernels` entry
binds a module/qualified symbol to a source file and hash. Target adapters
handle target-specific specialization and runtime records; generic replay does
not infer them from names.

Generic ownership certificates require a source file but do not require a
license or notice entry. The supplied target adapters additionally enforce their
pinned attribution files. The separate source-evidence format requires both a
source entry and a license or notice entry.

## Packet layout

```text
<packet>/
  certificate.json
  candidate.capture.json
  control.capture.json
  runtime.json
  benchmark.json
  controls.json
  reproducer.py
  external/<namespace>/tree/<declared source and attribution files>
```

`control.capture.json` always exists and canonically equals the embedded control,
including `null` for candidate-only packets. External files must be declared,
relative, unique, regular, within `external/`, and equal to their recorded size
and hash. Extra/missing files, symlinks, traversal and inconsistent kernel-source
bindings are rejected. Reproducer bytes are explicitly supplied to the writer
and bounded to 4 KiB.

Generated output uses `<output>/<target>/` packet roots. The checked examples
are `bundle/liger/liger-rmsnorm/` and `bundle/art/openpipe-art-local-stats/`.
`bundle/triton-layernorm/` contains the source and license files used for
inspection, not a packet. The bundle's
holdout, review and manifest JSON files are support records, not replay packets.
Discovery of packet directories is bounded and rejects ambiguous/incomplete
layouts.
Discovery rejects hidden child directories matching `.*.backup-*` or
`.*.staging-*` before traversing them; recover that residue before retrying an
audit. Regular files with those names remain ordinary root entries. Explicitly
selecting a packet or file checks its supplied bytes conditionally; it does not
authenticate that the packet is the current audit output.
Kernel function slices are limited to AST depth 128, 4,096 statements, and
16,384 AST nodes in both certificate and source-evidence replay.
Single-target replacement preserves valid siblings; promotion and cleanup
failure behavior is specified in [RUNTIME.md](RUNTIME.md).

Source evidence is a separate schema, `daggertrace.aot-source-evidence/2`, with
`source-evidence.json` and its declared external files. It represents one
named, decorated top-level kernel-function slice; generic replay does not prove
that a caller reaches that function. The pinned LayerNorm inspector follows both
reachable backward kernels and reports their classification without materializing
this packet; the source-evidence format can still select one named function from
a file containing several functions. The packet's `analysis` member has this
shape:

```json
{
  "outcome": "UNKNOWN",
  "reasons": ["MEMORY_MODEL"],
  "borrowed_writes": [],
  "collapse": null,
  "unknown_detail": "UNKNOWN:MEMORY_MODEL: source contains a Triton memory protocol operation"
}
```

## Replay trust boundary

Generic certificate replay checks schema/resource bounds, canonical digests,
capture files, the declared file set, graph and role checks, kernel bindings, and
the supplied source/HOP bindings against the stored access-role summary. It
does not regenerate TTIR. Replay then reinterprets storage/contribution
transitions and the paired-control relation, and compares the derived result
with the record. Source-evidence replay checks its declared files,
bounded function slice, pointer roles, and required `UNKNOWN:MEMORY_MODEL`
result; it has no AOT graph, capture files, or TTIR to replay. The CLI
additionally runs the target-specific checks for the known certificate target.

For a generic ownership certificate, `VALID` means all replay checks completed,
including checker-source compatibility. The checker source must
match the installed source manifest and production line count, or an explicitly
listed historical compatibility identity; local packet consistency alone is
insufficient. Source-evidence replay does not perform the checker-source check.

Replay has a separate semantic implementation and does not call the producer's
model or analyzer. Using the same protocol vocabulary does not authenticate the
observations in a packet. Replay neither fetches upstream source, authenticates
a Git revision, reruns AOTAutograd/Inductor, executes CUDA, nor proves gradient
values. A successful replay is a conditional judgment about the supplied
capture and source bytes.

The checked packets are historical: Liger at
`e5164336825e34d0161d762c294e25d9f35649cc` records a constructed residual's
`VIOLATION` + `CONTRIBUTION_COLLAPSE`; ART at
`7585dda17553bd7996080fe645229b31011758cf` records candidate-only
`SAFE_IN_FRAGMENT`. LayerNorm inspection at
`f797708c0626e5f9840ca5b0a98790e2c7cb09ad` records
`UNKNOWN:MEMORY_MODEL`. These records cover different evidence types.
