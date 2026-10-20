# Checked evidence bundle

The bundle contains two historical v2 replay packets, the source files used by
one source-only `inspect` run, and support records. The packets are replay
examples, not current GPU validation.

```text
bundle/
  liger/liger-rmsnorm/
    certificate.json
    candidate.capture.json
    control.capture.json
    runtime.json
    benchmark.json
    controls.json
    reproducer.py
    external/...
  art/openpipe-art-local-stats/
    certificate.json
    candidate.capture.json
    control.capture.json
    runtime.json
    benchmark.json
    controls.json
    reproducer.py
    external/...
  triton-layernorm/external/...
  holdout.json
  review.json
  manifest.json
```

The Liger packet reports `VIOLATION` with
`BORROWED_COTANGENT_WRITE` and `CONTRIBUTION_COLLAPSE`. The ART packet has a
null control and reports `SAFE_IN_FRAGMENT` with a fresh destination; no
supported HOP write reaches the modeled borrowed incoming-gradient storage. The
Triton tutorial source is classified `UNKNOWN:MEMORY_MODEL` because its atomic
lock, read-modify-write, and barrier protocol is outside the supported memory
model; it is available through the source-only `inspect` command and is not
represented as an AOT certificate.

`holdout.json` is an archival v2 record, not current release evidence or a replay
packet; its exact historical source files are unavailable. `review.json` and
`manifest.json` are likewise support records. `manifest.json` records the size
and SHA-256 of every sibling evidence file except itself. Each certificate also
carries a target inventory stored in the packet and its own canonical digest. These
hashes establish equality to supplied bytes, not remote provenance.

```bash
PYTHONPATH=src python -B -m daggertrace replay \
  evidence/results/bundle/liger/liger-rmsnorm

PYTHONPATH=src python -B -m daggertrace replay \
  evidence/results/bundle/art/openpipe-art-local-stats
```

Replay validates the packet's declared files, normalized capture semantics, source and effect
bindings, ownership, and the recorded result without executing PyTorch, Triton,
Inductor, or CUDA. Packet details are specified in
[`../../docs/CERTIFICATES.md`](../../docs/CERTIFICATES.md).
