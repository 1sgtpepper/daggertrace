# Generated packets

Live audits write packets beneath `artifacts/final/<target>/`. Generated runtime
output is ignored by Git. The [checked historical bundle](../evidence/results/bundle/)
is retained separately.

A packet contains its certificate, candidate and control capture files, supporting
runtime, benchmark, and control records, a reproducer, and declared external
source and attribution files. Candidate-only packets retain a canonical control
file containing `null`. Independent replay validates these bytes and the recorded ownership
judgment; it does not execute the reproducer or authenticate remote provenance.

```bash
daggertrace replay artifacts/final/openpipe-art-local-stats
```

The exact layout and integrity rules are in
[Certificates](../docs/CERTIFICATES.md); commands are in
[Runtime](../docs/RUNTIME.md).
