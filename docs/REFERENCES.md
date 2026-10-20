# References and provenance

## Framework contracts

1. [PyTorch: Extending autograd](https://docs.pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd)
   defines mutation declarations, saved tensors, and backward contracts.

2. [PyTorch 2.12: `torch.library.triton_op`](https://docs.pytorch.org/docs/2.12/library.html#torch.library.triton_op)
   defines compiler-visible registration for custom Triton operators.

3. [PyTorch custom Triton kernel tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)
   documents the `torch.compile` integration path.

4. The pinned
   [functional Triton HOP implementation](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/torch/_higher_order_ops/triton_kernel_wrap.py)
   is the framework path normalized by DaggerTrace capture.

5. [AOTAutograd architecture](https://docs.pytorch.org/functorch/stable/notebooks/aot_autograd_optimizations.html)
   describes the forward/backward graph boundary used by the checker.

## Pinned target provenance

| Target | Revision | Included files |
| --- | --- | --- |
| [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/tree/e5164336825e34d0161d762c294e25d9f35649cc) | `e5164336825e34d0161d762c294e25d9f35649cc` | RMSNorm source, utility dependencies, top-level BSD-2-Clause license, third-party Apache-2.0/MIT licenses, NOTICE, Unsloth attribution |
| [OpenPipe ART](https://github.com/OpenPipe/ART/tree/7585dda17553bd7996080fe645229b31011758cf) | `7585dda17553bd7996080fe645229b31011758cf` | `topk.py`, `LICENSE`, `THIRD-PARTY-NOTICES`, GPL-3.0 and LGPL-3.0 texts |
| [Triton LayerNorm tutorial](https://github.com/triton-lang/triton/blob/f797708c0626e5f9840ca5b0a98790e2c7cb09ad/python/tutorials/05-layer-norm.py) | `f797708c0626e5f9840ca5b0a98790e2c7cb09ad` | tutorial source and MIT license |

The manifests stored in each packet contain the exact sizes and SHA-256 values. Repository
links and revisions are provenance metadata; packet hashes establish equality
only to the supplied bytes.

## Language and verification context

6. Philippe Tillet, H. T. Kung, and David Cox,
   [“Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations”](https://doi.org/10.1145/3315508.3329973),
   MAPL 2019.

7. Adam Betts et al.,
   [“The Design and Implementation of a Verification Technique for GPU Kernels”](https://doi.org/10.1145/2743017),
   TOPLAS 2015. GPUVerify addresses race and barrier properties; DaggerTrace
   checks a distinct whole-storage ownership property.

8. William S. Moses et al.,
   [“Reverse-Mode Automatic Differentiation and Optimization of GPU Kernels via Enzyme”](https://doi.org/10.1145/3458817.3476165),
   SC 2021. Enzyme synthesizes derivatives; DaggerTrace checks captured custom
   backward kernels.

9. [NVIDIA PTX memory consistency model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model)
   defines ordering and scope semantics for atomic and barrier operations,
   outside DaggerTrace's supported memory model.
