"""Independent reduction holdout for the local evidence collector."""

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 8192


@triton.jit
def reduction_forward_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(values, axis=0)
    tl.store(y_ptr, total)


@triton.jit
def reduction_backward_kernel(dy_ptr, dx_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_elements
    dy = tl.load(dy_ptr)
    tl.store(dx_ptr + offsets, dy, mask=mask)


class ReductionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.device.type != "cuda":
            raise ValueError("reduction holdout requires a CUDA tensor")
        if x.ndim != 1 or not x.is_contiguous():
            raise ValueError("reduction holdout requires a contiguous rank-one tensor")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("reduction holdout requires float16, bfloat16, or float32")
        if not 1 <= x.numel() <= BLOCK_SIZE:
            raise ValueError(f"reduction holdout length must be between 1 and {BLOCK_SIZE}")
        ctx.n_elements = x.numel()
        y = torch.empty(1, device=x.device, dtype=x.dtype)
        reduction_forward_kernel[(1,)](
            x, y, ctx.n_elements, BLOCK=BLOCK_SIZE, num_warps=4
        )
        return y

    @staticmethod
    def backward(ctx, dy):
        dx = torch.empty(ctx.n_elements, device=dy.device, dtype=dy.dtype)
        reduction_backward_kernel[(1,)](
            dy, dx, ctx.n_elements, BLOCK=BLOCK_SIZE, num_warps=4
        )
        return dx


def apply(x):
    return ReductionFunction.apply(x)
