# Changelog

## Unreleased

- Focus the public alpha on live cotangent-storage ownership analysis, independent
  replay, Liger/ART runtime records and source-only LayerNorm inspection.
- Retire offline-v3 capture/certificate support, `seal`/`ingest`, CLI replay
  without target-specific checks, the project-owned calibrated-softmax target and standalone
  performance/profiler collectors. Historical `/2` packet replay remains supported.
- Clarify the constructed Liger residual, conditional ownership verdicts, and the
  difference between packet consistency and runtime correctness.
- Remove unused private exports, the private `Analysis._capture_schema` constructor
  field, and the undocumented production-review collector.
- Apply canonical capture validation to direct capture objects, clean up failed source
  loads, and enforce source AST bounds before replay effect analysis.
- Tighten holdout isolation from target adapters and manifest path checks; support holdout
  provenance from source archives through their `SOURCE_REVISION` marker.
- Remove dynamic private validation imports and Liger source-loading/staging
  forwarding layers; validate test fixtures against the checked source bytes.
- Classify unsupported premises by diagnostic prefixes rather than reason words
  appearing in user-supplied identifiers.
- Reject incomplete certificate sections before staging and distinguish completed
  audit promotion from subsequent backup-cleanup failure.
- Define analyzer rejection/`UNKNOWN` boundaries, checker-identity checks,
  stored access-summary replay, and handled CLI exit statuses precisely.

- Remove private facade forwarding, unused Liger reference modes, and duplicate
  grid validation; retain public object identities and independent replay.
- Validate backward saved-output roles against their separate forward domain
  consistently in ownership analysis and replay.
- Preserve primary target-cache failures when staging cleanup also fails; report
  audit cleanup failures independently of an active caller exception.
- Reject retained transaction directories on audit replacement and packet-root
  discovery; identify semantic-oracle cases by their varying inputs.
- Specify the backward twin comparison and source memory-model classification
  precisely; remove the atomic-replacement claim from audit help.

- Normalize recursive capture dictionaries to `InvalidArtifact` and retain the
  discovered packet layout when formatting replay results.
- Remove unused ART return transport, the private unpinned module-load mode,
  redundant replay/oracle checks, and duplicate test graph constructors.

- Reject non-finite numerical results and negative absolute-error deltas in
  the Liger and ART examples, including their compiled comparisons.
- Keep test-only semantic projections outside the runtime and clarify replay
  target selection and successful source-evidence exit status.
- Use plain language in public documentation and command help.

## 0.4.0 — 2026-08-17

- Reframed DaggerTrace as a target-parameterized checker over a closed,
  versioned PyTorch 2.12.1/Triton 3.7.1 ownership language at the
  AOTAutograd/Triton boundary.
- Separated the target-neutral normalized capture, whole-storage ownership model,
  and independent offline replay from embedded certificate target manifests and
  online adapters.
- Made the embedded certificate target manifest the owner of exact source files, source/license
  provenance, kernel source bindings, specialization, runtime/device constants, and
  theorem constants. Normalized captures retain literal graph structure and dense
  HOP node IDs for generic replay.
- Defined nullable controls: candidate-only packets can prove `VIOLATION` or
  `SAFE_IN_FRAGMENT`, while contribution collapse requires a valid paired control. Every
  certificate packet retains `control.capture.json`; candidate-only certificate packets use canonical
  JSON `null` plus one trailing newline.
- Recorded the three evidence boundaries: Liger `VIOLATION` plus collapse, pinned
  OpenPipe ART `topk.py` candidate-only `SAFE_IN_FRAGMENT`, and the official Triton
  LayerNorm tutorial source classification `UNKNOWN` with reason `MEMORY_MODEL`.
  A `source-evidence.json` packet contains one named kernel-function slice;
  the LayerNorm inspector follows both reachable backward kernels in the pinned
  tutorial and reports their classification without materializing that packet.
  The wire value is `UNKNOWN:MEMORY_MODEL`; the CLI renders it as
  `UNKNOWN: MEMORY_MODEL`.
- Added the ART source profile for `tl.exp`, `tl.max`, `tl.static_range`,
  `tl.where`, bare `float`, `aten.empty_like`, and `aten.zeros`. The pinned file
  contains three direct `@triton.jit` launches with `num_warps=8` and no
  `@triton.autotune` or `triton.Config`; that launch shape remains target evidence.
- Distinguished source decoration from the observed ART runtime capture:
  three HOPs, each with an `Autotuner` over `JITFunction` wrapper chain, the
  exact config `num_warps=8`, `num_ctas=1`, `num_stages=3`, and a distinct
  `jit_cache_key`. Those values belong to the ART target manifest, not generic
  replay.
- Distinguished AOTAutograd and Inductor occurrence evidence from claims about
  lowering or verifying an external Triton body.
- Defined `/2` capture, certificate, source-evidence, and profile identities;
  target-specific identity remains embedded in certificate `target` data rather
  than in the generic core/replayer contract.
- Clarified that dependency-light packet replay validates supplied packet bytes and capture
  semantics, not remote Git authenticity.
- Retained the explicit no-arbitrary-Triton boundary and the distinction between
  source/effect evidence, ownership results, and runtime occurrence evidence.
