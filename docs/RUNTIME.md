# Runtime commands

## Audit and reproduce

Online audit requires the [pinned GPU environment](ENVIRONMENT.md). The live
targets are `liger-rmsnorm` and `openpipe-art-local-stats`:

```bash
daggertrace audit liger-rmsnorm --gpu \
  --target-root <pinned-liger-root> --output artifacts/final
daggertrace replay artifacts/final/liger-rmsnorm
daggertrace reproduce liger-rmsnorm --gpu --target-root <pinned-liger-root>
```

Audit validates selected regular source bytes and their hashes, loads the target,
captures AOT forward/backward graphs, compares source-derived effects with the
captured TTIR access-role summary, checks HOP bindings, runs ownership analysis
and the numerical comparisons, then writes and independently replays the packet.
An absent audit `--target-root` permits fetching the pinned external source into
`<output-parent>/.target-cache`. Reproduction always requires an explicit target
root.

`audit all` and `reproduce all` select both certificate targets. With
`--target-root <root>`, their checkouts must be `<root>/liger-rmsnorm/` and
`<root>/openpipe-art-local-stats/`. A single-target update replaces its selected
child packet and preserves valid siblings. Results for multiple targets replace
the previous output together. Errors before replacement preserve the existing
output. If replacement fails, audit attempts restoration; a restoration failure
reports the backup directory. If cleanup fails after replacement, the new output remains installed
and the error reports the remaining backup or staging directories. A retry refuses hidden
backup or staging directories matching `.<output-name>.backup-*` or
`.<output-name>.staging-*` until the leftover directories are resolved; unrelated
regular files with those names are preserved. Per-target staging beside the
root uses `.<output-root>.<target>.staging-*` and is refused before a new
packet staging directory is allocated. Replacing the entire output directory
also rejects immediate child backup or staging directories before moving the
previous root. Replay
rejects those leftover backup or staging directories while searching for packets.
Explicitly selecting a packet or file checks its supplied bytes; it does not
establish that the packet is the latest audit output.

All numerical differences must be finite. Comparisons expected to agree require
a nonnegative maximum absolute difference within tolerance. The Liger gradient
error and incoming-gradient mutation must exceed their thresholds.

The Liger adapter constructs a residual and compares writes to borrowed storage
with writes to a new buffer. This demonstrates the failure at a pinned revision;
it does not establish whether current upstream Liger is affected. ART is a candidate-only ownership example.
Selected source bytes are verified; repository URLs and revisions are provenance
metadata and do not authenticate a remote checkout.

## Source-only inspection

```bash
daggertrace inspect triton-layernorm --target-root <pinned-triton-root>
```

The pinned tutorial reaches two backward kernels and a synchronization protocol
outside the supported memory model. Inspection reports `UNKNOWN:MEMORY_MODEL`;
it does not write an AOT certificate or run the target.

## Replay

```bash
daggertrace replay evidence/results/bundle/liger/liger-rmsnorm
daggertrace replay evidence/results/bundle/art/openpipe-art-local-stats --json
```

Replay accepts a certificate or source-evidence file, a packet directory, or a
directory containing packets, subject to size and depth limits. Certificate
replay checks declared files, source bindings, graph semantics, stored access-role
data, compatibility with the checker source, and the recorded ownership result; it does not regenerate TTIR. Source-evidence replay
checks its declared source files, bounded function slice, pointer roles, and
required `UNKNOWN:MEMORY_MODEL` result. Neither path imports PyTorch/Triton,
executes a reproducer, or accesses the network. The CLI then runs the
target-specific checks for supported certificates. For a generic ownership
certificate, `VALID` means all replay checks passed, not that the kernel is
generally correct; source-evidence replay does not check the checker source.

`runtime.json`, `benchmark.json`, and other supporting records are checked for
internal consistency. Their presence does not establish a new execution or
measurement. Historical packets remain historical when replayed by new code.
The separate offline-v3 seal/ingest workflow is retired.

Generated packet layout and trust limits are specified in
[CERTIFICATES.md](CERTIFICATES.md).

## Exit status

Exit `0` means the command completed and emitted its result, including source
inspection with `UNKNOWN`, source-evidence replay with `UNKNOWN:MEMORY_MODEL`,
and certificate replay with `SAFE_IN_FRAGMENT` or `VIOLATION`.
Exit `1` is emitted when handled replay validation raises `ReplayError`,
including failures in packet discovery, file validation, or target-specific checks. Exit `2` is emitted for a
handled `DaggerTraceError` such as argument, target, environment, or audit
setup/execution failure; argument-parser errors also use `2`. Exit `130` means
the command was interrupted. Unexpected programming exceptions are not
normalized by this CLI.
