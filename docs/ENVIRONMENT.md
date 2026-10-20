# Environment

## Replay and validation without a GPU

Python 3.10–3.14 supports packet replay and validation without PyTorch, Triton,
or CUDA:

```bash
python -m pip install -e '.[dev]'
make validate
```

## GPU capture and runtime environment

Capture and runtime checks require exactly:

```text
torch:     2.12.1+cu130
torch git: 7269437d655783a26cba32aa88195b741ff496aa
triton:    3.7.1
capture:   fullgraph=true, dynamic=false, post-dynamo-pre-inductor
```

Capture depends on private PyTorch APIs. Other framework versions are not
supported in this environment. Target evidence also records Python, CUDA
runtime/driver, device, launch settings and source identity; those observations
are not generalized to other systems.

Hosted CI has no CUDA runner. GPU validation must be run manually in the pinned
environment, for example:

```bash
DAGGERTRACE_LIGER_ROOT=<pinned-liger-root> \
make validate-gpu DAGGERTRACE_TARGET=liger-rmsnorm \
  DAGGERTRACE_TARGET_ROOT=<pinned-liger-root>
```

Set `DAGGERTRACE_ART_ROOT` for ART-specific GPU tests. External GPU tests are
skipped when their pinned checkout environment variable is absent; the selected
audit and reproducer still require a working GPU and the exact pinned source
tree. Retain the complete run record when making runtime claims. Passing CI
without CUDA does not establish GPU or compiler correctness.
