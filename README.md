# DaggerTrace

DaggerTrace checks one dangerous custom-backward pattern: writing a result into
storage borrowed from an incoming gradient. In a residual such as
`y = x + op(x)`, that write can erase the gradient contribution needed by the
identity branch.

It captures a restricted set of PyTorch AOTAutograd operations, including
functional Triton higher-order operators (HOPs). It saves the graphs, source
files, and ownership result in a packet that can be validated without a GPU.
**This is an alpha research checker.** The live CLI supports the Liger RMSNorm
and OpenPipe ART examples below; it does not accept arbitrary user kernels.

## Try it without CUDA

```bash
python -m pip install -e .
daggertrace replay evidence/results/bundle/liger/liger-rmsnorm
```

The included historical Liger packet replays as `VALID` and records
`VIOLATION` with `BORROWED_COTANGENT_WRITE` and `CONTRIBUTION_COLLAPSE`; see
the [checked evidence bundle](evidence/results/README.md).

For a generic ownership certificate, `VALID` means the packet passed validation.
Replay checks its format and resource limits, declared files and hashes, checker
source, and kernel bindings, then independently recomputes the ownership result.
It does not mean the kernel is correct. Replay does not run the target or
authenticate its Git revision.

## The failure being checked

```text
y = x + rms_norm(x)

borrowed destination: backward writes dX into incoming dY storage
                     -> VIOLATION + CONTRIBUTION_COLLAPSE
fresh destination:    backward writes dX into a new buffer
                     -> SAFE_IN_FRAGMENT
```

The Liger adapter constructs this residual context and selects in-place versus
out-of-place behavior at one pinned revision. This demonstrates the failure in
that context; it does not establish whether current upstream Liger is affected.
The control keeps the forward graph and backward kernel identity while changing destination
ownership.

## Results and limits

| Result | Meaning |
| --- | --- |
| `SAFE_IN_FRAGMENT` | No supported HOP write argument is bound to borrowed incoming-gradient storage in the captured graph. |
| `VIOLATION` | A supported HOP write argument is bound to that borrowed storage in the captured graph. |
| `UNKNOWN` | Analysis or source inspection could not establish a required assumption about the graph, source, aliasing, memory, or matching control; ownership certificates cannot store this outcome. |
| Replay rejection | The packet's format, files, hashes, checker identity, or recorded analysis failed validation. |

A paired control can additionally establish `CONTRIBUTION_COLLAPSE`: the
normalized backward graph replaces a live tangent contribution. A borrowed write
alone does not establish this stronger result. `SAFE_IN_FRAGMENT` proves neither
gradient values nor bounds, race freedom, barrier safety, or arbitrary compiler
lowering. See the [semantic contract](docs/SEMANTICS.md).

## Supported examples

| Example | Evidence |
| --- | --- |
| [Liger RMSNorm residual](evidence/results/bundle/liger/liger-rmsnorm/) | Historical paired `VIOLATION` and contribution-collapse packet. |
| [OpenPipe ART local statistics](evidence/results/bundle/art/openpipe-art-local-stats/) | Historical candidate-only `SAFE_IN_FRAGMENT` packet. |
| [Triton LayerNorm](evidence/results/bundle/triton-layernorm/) | Source-only `UNKNOWN:MEMORY_MODEL`; its synchronization protocol is outside the checker. |

Live audits require the exact PyTorch **2.12.1+cu130** revision and Triton
**3.7.1** profile in [Environment](docs/ENVIRONMENT.md). Hosted CI checks the
Python package, semantic cases, and GPU-free packet replay; it has no CUDA runner.
The checked packets are historical observations, not new executions of the
current checkout. No general coverage, adoption, or speedup is claimed.

## Live audit

On the pinned GPU environment, with an exact target checkout:

```bash
daggertrace audit liger-rmsnorm --gpu \
  --target-root <pinned-liger-root> --output artifacts/final
daggertrace replay artifacts/final/liger-rmsnorm
daggertrace reproduce liger-rmsnorm --gpu --target-root <pinned-liger-root>
```

A live audit normalizes AOT graphs, compares source reads and writes with the
captured Triton IR (TTIR) access summary, checks HOP argument bindings, analyzes
storage and contribution identity, runs the numerical comparisons, and writes
a replayable packet. The CLI also supports `openpipe-art-local-stats`;
`inspect triton-layernorm` classifies source code without running a kernel.

## Development

For validation setup, follow [Development and validation](docs/DEVELOPMENT.md)
to install the `[dev]` extra before running:

```bash
make validate
```

Python 3.10–3.14 is supported for replay and validation without PyTorch or Triton.
Generic replay remains separate from capture and ownership analysis;
certificate CLI replay also runs target-specific checks for supported examples
and rejects unsupported repositories.

- [Runtime commands](docs/RUNTIME.md)
- [Certificate format and trust boundary](docs/CERTIFICATES.md)
- [Development and validation](docs/DEVELOPMENT.md)
- [Evidence inventory](evidence/README.md)
- [References and source attribution](docs/REFERENCES.md)
