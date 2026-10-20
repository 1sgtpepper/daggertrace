# Evidence

The checked bundle contains historical Liger and ART ownership packets, Triton
LayerNorm source files for inspection, and support records. Its bytes are fixed;
current validation does not relabel them as new executions.

| Artifact | Meaning |
| --- | --- |
| [Liger packet](results/bundle/liger/liger-rmsnorm/) | Constructed residual at one pinned revision: borrowed write and paired contribution collapse. |
| [ART packet](results/bundle/art/openpipe-art-local-stats/) | Candidate-only `SAFE_IN_FRAGMENT`, with null control. |
| [LayerNorm source files](results/bundle/triton-layernorm/) | Source-only `UNKNOWN:MEMORY_MODEL`; no AOT packet. |
| [Review record](results/bundle/review.json) | Historical 48 semantic cases, 34 packet mutations, 7 ablations, relocation and subprocess replay. |
| [Holdout record](results/bundle/holdout.json) | Archival record; its historical source files are unavailable, so it is not current release evidence. |
| [Manifest](results/bundle/manifest.json) | Size and SHA-256 inventory of sibling files, not remote provenance. |

When collected from a source archive, the holdout records the exact lowercase
40-hex `SOURCE_REVISION` marker. Git metadata is absent in that mode, so
`source_tree_dirty` is reported as unknown.

Current validation without a GPU checks the retained v2 semantic cases and
ablations and replays both packets. The mutation count in a packet is a recorded
result; ordinary replay does not rerun the historical mutation collection.

The holdout collector does not import a supported target adapter. It loads three
held-out Triton modules and executes them on the exact pinned GPU profile described
in [Environment](../docs/ENVIRONMENT.md). The review collector reads supplied Liger
and ART packets and a pinned Triton source tree.

```bash
PYTHONPATH=src python -B evidence/collect_holdout.py \
  --output evidence-output/holdout.json --sizes 257 4097

PYTHONPATH=src python -B evidence/collect_review.py \
  --liger evidence/results/bundle/liger/liger-rmsnorm/certificate.json \
  --art evidence/results/bundle/art/openpipe-art-local-stats/certificate.json \
  --triton-root <pinned-triton-root> --output evidence-output/review.json
```

The ART and Liger source, notice, and license files are retained in their packets.
Hashes bind supplied bytes and do not authenticate upstream history. No speedup
claim is made from the unavailable historical timing/profiler archive; the
standalone performance collectors have been removed.

See [Certificates](../docs/CERTIFICATES.md) and [Runtime](../docs/RUNTIME.md) for
replay and collection boundaries.
