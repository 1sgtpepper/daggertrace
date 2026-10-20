from __future__ import annotations

import pytest

from daggertrace.capture import _kernel_runtime
from daggertrace.errors import UnsupportedCapture


triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")


@triton.jit
def _unit_kernel(X_ptr, Y_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(Y_ptr + offsets, tl.load(X_ptr + offsets))


def _autotuner(*, configs=None, **kwargs):
    return triton.autotune(
        configs=configs or [triton.Config({}, num_warps=4, num_stages=3)],
        key=[],
        **kwargs,
    )(_unit_kernel)


def _autotuner_with_config_mutation(field, value):
    wrapper = _autotuner()
    setattr(wrapper.configs[0], field, value)
    return wrapper


def test_kernel_runtime_binds_exact_single_config_and_jit_cache_key():
    for num_warps in (1, 2, 4, 8, 16, 32):
        wrapper = _autotuner(configs=[triton.Config({}, num_warps=num_warps, num_stages=3)])
        runtime, function = _kernel_runtime(wrapper)
        assert function is _unit_kernel.fn
        assert runtime == {
            "wrapper_type": "triton.runtime.autotuner.Autotuner",
            "leaf_type": "triton.runtime.jit.JITFunction",
            "config": {
                "kwargs": {},
                "num_warps": num_warps,
                "num_ctas": 1,
                "num_stages": 3,
                "maxnreg": None,
                "pre_hook": None,
                "ir_override": None,
            },
            "jit_cache_key": _unit_kernel.cache_key,
        }


@pytest.mark.parametrize(
    ("num_ctas", "num_stages"),
    [(2, 3), (1, 1), (1, 32)],
)
def test_kernel_runtime_binds_bounded_nonfixture_config_counts(num_ctas, num_stages):
    wrapper = _autotuner(
        configs=[triton.Config({}, num_warps=8, num_ctas=num_ctas, num_stages=num_stages)]
    )

    runtime, function = _kernel_runtime(wrapper)

    assert function is _unit_kernel.fn
    assert runtime["config"]["num_ctas"] == num_ctas
    assert runtime["config"]["num_stages"] == num_stages


@pytest.mark.parametrize(
    "kernel, detail",
    [
        (_unit_kernel, "Autotuner -> JITFunction"),
        (
            _autotuner(
                configs=[
                    triton.Config({}, num_warps=4),
                    triton.Config({}, num_warps=8),
                ]
            ),
            "exactly one exact Config",
        ),
        (_autotuner_with_config_mutation("kwargs", []), "exact admitted profile"),
        (_autotuner_with_config_mutation("kwargs", {"BLOCK_SIZE": 128}), "exact admitted profile"),
        (_autotuner_with_config_mutation("num_warps", 0), "exact admitted profile"),
        (_autotuner(configs=[triton.Config({}, num_warps=3, num_stages=3)]), "exact admitted profile"),
        (_autotuner_with_config_mutation("num_warps", 33), "exact admitted profile"),
        (_autotuner_with_config_mutation("num_warps", 64), "exact admitted profile"),
        (_autotuner_with_config_mutation("num_warps", True), "exact admitted profile"),
        (_autotuner_with_config_mutation("num_ctas", 0), "exact admitted profile"),
        (_autotuner_with_config_mutation("num_ctas", 33), "exact admitted profile"),
        (_autotuner_with_config_mutation("num_ctas", True), "exact admitted profile"),
        (_autotuner_with_config_mutation("num_stages", 0), "exact admitted profile"),
        (_autotuner_with_config_mutation("num_stages", 33), "exact admitted profile"),
        (_autotuner_with_config_mutation("num_stages", True), "exact admitted profile"),
        (_autotuner_with_config_mutation("num_stages", 3.0), "exact admitted profile"),
        (_autotuner_with_config_mutation("pre_hook", "hook"), "exact admitted profile"),
        (_autotuner_with_config_mutation("ir_override", "override"), "exact admitted profile"),
        (_autotuner(reset_to_zero=["X_ptr"]), "hooks/reset/restore"),
    ],
)
def test_kernel_runtime_rejects_other_wrapper_states(kernel, detail):
    with pytest.raises(UnsupportedCapture, match=detail):
        _kernel_runtime(kernel)
