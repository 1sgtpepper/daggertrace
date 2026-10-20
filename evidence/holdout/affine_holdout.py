"""Independent two-input Triton holdout for the local evidence collector."""

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 128


@triton.jit
def affine_forward_kernel(x_ptr, w_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, x * w + 0.25, mask=mask)


@triton.jit
def affine_backward_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    dx_ptr,
    dw_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
    tl.store(dx_ptr + offsets, dy * w, mask=mask)
    tl.store(dw_ptr + offsets, dy * x, mask=mask)


class AffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        n_elements = x.numel()
        y = torch.empty_like(x)
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        affine_forward_kernel[grid](
            x, weight, y, n_elements, BLOCK=BLOCK_SIZE, num_warps=4
        )
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        n_elements = x.numel()
        dx = torch.empty_like(x)
        dweight = torch.empty_like(weight)
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        affine_backward_kernel[grid](
            x,
            weight,
            dy,
            dx,
            dweight,
            n_elements,
            BLOCK=BLOCK_SIZE,
            num_warps=4,
        )
        return dx, dweight


def apply(x, weight):
    return AffineFunction.apply(x, weight)
