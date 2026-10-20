"""Held-out differentiable Triton kernel used by the local evidence collector.

This module is intentionally outside ``src/daggertrace``.  The collector loads it
after freezing the checker commit, so the holdout exercises the existing capture and
ownership language without adding a target adapter or a target name to the checker.
"""

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 128


@triton.jit
def holdout_forward_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, values + 1.0, mask=mask)


@triton.jit
def holdout_backward_kernel(dy_ptr, dx_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
    tl.store(dx_ptr + offsets, values, mask=mask)


@triton.jit
def composed_forward_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, values * 1.5 + 0.25, mask=mask)


@triton.jit
def composed_backward_kernel(dy_ptr, dx_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
    tl.store(dx_ptr + offsets, values * 1.5, mask=mask)


class HoldoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.n_elements = x.numel()
        y = torch.empty_like(x)
        grid = (triton.cdiv(ctx.n_elements, BLOCK_SIZE),)
        holdout_forward_kernel[grid](
            x, y, ctx.n_elements, BLOCK=BLOCK_SIZE, num_warps=4
        )
        return y

    @staticmethod
    def backward(ctx, dy):
        dx = torch.empty_like(dy)
        grid = (triton.cdiv(ctx.n_elements, BLOCK_SIZE),)
        holdout_backward_kernel[grid](
            dy, dx, ctx.n_elements, BLOCK=BLOCK_SIZE, num_warps=4
        )
        return dx


def apply(x):
    return HoldoutFunction.apply(x)


# The two residual twins differ only in destination ownership; the unsafe branch is
# a deliberate negative control for borrowed-cotangent invalidation.
class ComposedHoldoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, in_place):
        ctx.n_elements = x.numel()
        ctx.in_place = in_place
        y = torch.empty_like(x)
        grid = (triton.cdiv(ctx.n_elements, BLOCK_SIZE),)
        composed_forward_kernel[grid](
            x, y, ctx.n_elements, BLOCK=BLOCK_SIZE, num_warps=4
        )
        return y

    @staticmethod
    def backward(ctx, dy):
        dx = dy if ctx.in_place else torch.empty_like(dy)
        grid = (triton.cdiv(ctx.n_elements, BLOCK_SIZE),)
        composed_backward_kernel[grid](
            dy, dx, ctx.n_elements, BLOCK=BLOCK_SIZE, num_warps=4
        )
        return dx, None


def composed_unsafe(x):
    return x + ComposedHoldoutFunction.apply(x, True)


def composed_safe(x):
    return x + ComposedHoldoutFunction.apply(x, False)
