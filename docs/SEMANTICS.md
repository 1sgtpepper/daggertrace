# Semantic contract

DaggerTrace checks borrowed incoming-gradient storage in a restricted graph
language using PyTorch AOTAutograd and functional Triton higher-order operators
(HOPs). It does not synthesize derivatives or infer correctness from numerical
agreement.

## Property and scope

The checker models each supported output cotangent passed to a custom backward as
borrowed read-only storage. Within the supported language, no functional Triton
HOP write role may bind to storage carrying that borrowed tangent identity.
Physical byte mutation, functional replacement, and contribution liveness are
separate observations.

`SAFE_IN_FRAGMENT` means no supported HOP write violates this property.
`VIOLATION` records a borrowed write. For an exact v2 `Capture` that passes
type and schema checks, `UNKNOWN` means a required graph, source, alias,
memory, or paired-control premise cannot be established. At the analyzer
boundary, dictionary inputs are parsed by `Capture.from_dict`: malformed
dictionaries, unsupported root types, and unsupported schemas raise
`InvalidArtifact` before semantic analysis, including cyclic or excessively
recursive structures. An exact v2 `Capture` whose
structural, profile, or semantic validation fails inside the analyzer returns
an `Analysis` whose `outcome` is `Outcome.UNKNOWN`. Generic ownership replay
rejects malformed packet structure and persisted `UNKNOWN` records with
`ReplayError`; source evidence has a separate schema for the
`UNKNOWN:MEMORY_MODEL` result. The result does not prove numerical correctness,
pointer bounds, partial alias ranges, race freedom, synchronization, or
general compiler correctness.

These results describe bindings in the captured graph. `VIOLATION` does not by
itself claim that a GPU instruction executed or that particular bytes were
mutated.

## Capture boundary

The checker accepts `daggertrace.aot-capture/2` under profile
`torch-2.12-triton-hop-whole-storage/2`. Certificates use
`daggertrace.cotangent-ownership/2`; source evidence uses
`daggertrace.aot-source-evidence/2`.

Live capture requires PyTorch 2.12.1+cu130 at
`7269437d655783a26cba32aa88195b741ff496aa` and Triton 3.7.1. It observes one
forward and one backward AOT callback after Dynamo and before Inductor, with
`fullgraph=true` and `dynamic=false`. Compiler activity and numerical comparisons
are supporting runtime records; they do not establish how the graph was lowered.

The public audit command selects the two supplied target adapters. Direct
`capture_aot` integration uses the same supported profile. Offline-v3 artifacts
and boundary-ownership sealing are no longer supported.

## Supported graph operations

- Forward placeholders carry `primal` input roles. In a backward graph, a
  `tangent` role names a forward `output` slot; explicit `primal` and
  `saved_output` roles must name their corresponding forward domains. Those
  roles are non-tangent inputs, while a missing role is treated as a saved
  value. Graph-output descriptors are separate: forward outputs use `output`
  or `saved_output`, and backward outputs use `gradient` roles pointing to
  `primal` slots.
- `aten.view`, `aten._unsafe_view`, `aten.alias`, and `aten.detach`.
- `aten.empty`, `aten.empty_like`, `aten.zeros`, `aten.zeros_like`, and `aten.clone`.
- `aten.mul`, `aten.sum`, and exact two-operand `aten.add`.
- Functional Triton HOPs and `operator.getitem` over their write results.
- One final output node.

Node IDs are dense and topological; references point backward. Marked gradient
accumulation is supported only on the exact add form. Other graph effects produce
`UNKNOWN:GRAPH_EFFECT`.

Tensor metadata records shape, stride, dtype, device, storage offset, and a
capture-local whole-storage `alias_group`. Views preserve storage group and
logical version. Allocations, `aten.clone`, `aten.mul`, `aten.sum`, `aten.add`,
and `operator.getitem` results over HOP writes receive fresh logical storage.
A functional HOP write result also receives a fresh logical result
state/version after the write. Its destination is the bound input state checked
before the write; a fresh result does not make a borrowed destination safe.
Clones preserve contribution parts but clear borrowed status. Contradictory
storage roots produce `UNKNOWN:REGION_ALIAS`. This abstraction does not establish
byte-range aliasing or global lifetime ownership.

## Triton effects and identity

Each HOP records its kernel, tensor bindings, writes, constants, and launch grid.
Pointer bindings exactly cover the read/write roles; tensor bindings and scalar
constants partition the formal arguments. Write destinations are distinct.
Kernel identity preserves source hash, formal roles, wrapper/JIT leaf types,
launch configuration, and JIT cache key.

The source profile supports direct `tl.arange`, `exp`, `exp2`, `expand_dims`,
`join`, `load`, `log2`, `max`, `program_id`, `split`, `static_range`, `store`,
`sum`, `where`, and `zeros` calls, supported `.to` conversions, and unshadowed
bare `float`, `min`, `range`, and `rsqrt`. It follows lexical pointer aliases
and supports only the `if` conditions and bounded loop syntax defined in
`effects.py`. Predicates may combine literals and constexpr-dependent
expressions. A module constant is allowed only in a two-operand comparison whose
other operand is constexpr-dependent; a bare module-constant predicate is not
supported. Pointer aliases introduced in a branch or loop do not escape it.

Loads and stores use only the supported positional arguments and mask forms. Unsupported calls,
control, ambiguous roots, imports, nested definitions, and asynchronous effects
are rejected from the source profile. Direct `tl.atomic_add`, `tl.atomic_and`,
`tl.atomic_cas`, `tl.atomic_max`, `tl.atomic_min`, `tl.atomic_or`,
`tl.atomic_xchg`, `tl.atomic_xor`, `tl.barrier`, and `tl.debug_barrier`
operations are classified as `UNKNOWN:MEMORY_MODEL`; arbitrary
namespace-qualified lock calls remain outside the supported source subset and are
classified as `UNKNOWN:SOURCE_PROFILE`. Certificates cannot be created for
unsupported source.

During live capture, source-derived reads and writes must agree with the
captured TTIR access summary, and HOP writes must agree with both. Replay checks
the supplied source and HOP bindings against the stored access summary; it does
not regenerate TTIR. These checks cover access roles, not arbitrary address
arithmetic, mask correctness, or executed machine instructions.

## Contribution analysis

Each value carries a storage group, logical version, contribution multiset,
borrowed tangent identity, and producer. Incoming tangents seed borrowed
identities. Marked gradient accumulation concatenates contributions; supported
arithmetic produces derived contributions. This tracks identity and
multiplicity, not algebraic equivalence or floating-point values.

`CONTRIBUTION_COLLAPSE` requires a paired control with the same framework,
capture configuration, forward graph, kernel identities, and sequence of
reachable backward operations after normalization. Forward graphs must match
exactly, including their alias labels and metadata. For the backward comparison,
storage `alias_group` labels are removed independently from each run, and every
node reference in `args` and `kwargs` is stripped. The comparison omits only
view, alias, and detach wrappers (`aten.view`,
`aten._unsafe_view`, `aten.alias`, `aten.detach`) and fresh allocation forms
(`aten.empty`, `aten.empty_like`, `aten.zeros`, `aten.zeros_like`). All
remaining reachable operator metadata—operation, target, stripped arguments and
keywords, role, tensor metadata after alias-group normalization,
`gradient_accumulation`, and output roles—must match, including the metadata for
`aten.clone` and `aten.add`. The graph comparison is separate from the
collapse proof: each run must produce the returned gradient state through a
marked gradient-accumulation `aten.add`, and the exact multiset relation must
show one live tangent contribution replaced while unrelated multiplicities
remain equal. That proof establishes the supported change from a borrowed
destination to a new buffer; other differences in the normalized graphs remain
unsupported.
Wrappers after that add are outside the supported collapse check and yield `UNKNOWN`.
Adding or replacing another reachable operation also makes the pair unsupported.

Candidate-only packets may establish a borrowed write or safe ownership, but
cannot establish contribution collapse. Replay independently re-derives the
same ownership result within these restrictions from supplied packet bytes. It
does not authenticate a live capture or rerun the GPU.
