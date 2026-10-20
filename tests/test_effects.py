from __future__ import annotations

import pytest

from daggertrace.effects import profile_triton_source
from daggertrace.errors import UnsupportedCapture


def test_profiles_alpha_renamed_pointer_aliases():
    source = """@triton.jit
def entirely_different_name(source, destination, stride):
    source_address = source + stride
    destination_address = destination + stride
    value = tl.load(source_address)
    tl.store(destination_address, value)
"""
    profile = profile_triton_source(source, {"source", "destination"})
    assert profile.reads == ("source",)
    assert profile.writes == ("destination",)
    assert [event.operation for event in profile.events] == ["tl.load", "tl.store"]


def test_atomic_is_rejected_at_the_memory_model_boundary():
    source = """@triton.jit
def accumulate(target, value):
    tl.atomic_add(target, value)
"""
    with pytest.raises(UnsupportedCapture, match="MEMORY_MODEL"):
        profile_triton_source(source, {"target"})


def test_ambiguous_pointer_root_is_unknown():
    source = """@triton.jit
def ambiguous(left, right):
    tl.store(left + right, 0.0)
"""
    with pytest.raises(UnsupportedCapture, match="formal roots"):
        profile_triton_source(source, {"left", "right"})


def test_unknown_effect_is_rejected():
    source = """@triton.jit
def opaque(pointer):
    tl.inline_asm_elementwise('x', '=r', [pointer], dtype=tl.int32, is_pure=False)
"""
    with pytest.raises(UnsupportedCapture, match="outside the supported source subset"):
        profile_triton_source(source, {"pointer"})


def test_unknown_helper_call_is_rejected():
    source = """@triton.jit
def opaque(pointer, output):
    value = evil(pointer)
    tl.store(output, value)
"""
    with pytest.raises(UnsupportedCapture, match="call evil is outside"):
        profile_triton_source(source, {"pointer", "output"})


def test_exact_pure_call_forms_remain_admitted():
    source = """@triton.jit
def copied(source, destination, width):
    start = tl.program_id(0).to(tl.int64)
    stop = min(start + 1, width)
    for index in range(start, stop):
        value = rsqrt(tl.load(source + index)).to(tl.float32)
        tl.store(destination + index, value)
"""
    profile = profile_triton_source(source, {"source", "destination"})
    assert profile.reads == ("source",)
    assert profile.writes == ("destination",)


def test_art_pure_call_forms_and_bare_float_remain_closed_and_admitted():
    source = """@triton.jit
def topk(source, destination):
    value = tl.load(source)
    value = tl.exp(value)
    peak = tl.max(value, axis=0)
    for _ in tl.static_range(0, 1):
        value = tl.where(value > peak, value, float("-inf"))
    tl.store(destination, value)
"""

    profile = profile_triton_source(source, {"source", "destination"})

    assert profile.reads == ("source",)
    assert profile.writes == ("destination",)


def test_base_2_math_and_joined_reduction_are_admitted_as_pure_operations():
    source = """@triton.jit
def backward(source, destination):
    value = tl.exp2(tl.load(source))
    normalizer = tl.log2(value)
    pairs = tl.join(value, value * normalizer)
    total, weighted = tl.split(tl.sum(pairs, axis=0))
    tl.store(destination, total + weighted)
"""

    profile = profile_triton_source(source, {"source", "destination"})

    assert profile.reads == ("source",)
    assert profile.writes == ("destination",)


def test_expand_dims_remains_a_pure_closed_shape_operation():
    source = """@triton.jit
def column_axis(source, destination):
    value = tl.load(source)
    value = tl.expand_dims(value, 0)
    tl.store(destination, value)
"""

    profile = profile_triton_source(source, {"source", "destination"})

    assert profile.reads == ("source",)
    assert profile.writes == ("destination",)


def test_atomic_protocol_is_rejected_as_a_memory_model_boundary():
    source = """@triton.jit
def protocol(lock, output):
    while True:
        old = tl.atomic_cas(lock, 0, 1)
    tl.debug_barrier()
    tl.atomic_xchg(lock, 0)
    tl.store(output, old)
"""

    with pytest.raises(UnsupportedCapture, match="MEMORY_MODEL"):
        profile_triton_source(source, {"lock", "output"})


@pytest.mark.parametrize(
    "body",
    [
        """        value = tl.load(pointer)
        tl.store(output, value)
        break
""",
        """        value = unknown(pointer)
        tl.store(output, value)
        break
""",
    ],
    ids=["pure-load-store", "unknown-call"],
)
def test_while_without_a_memory_protocol_is_source_profile(body):
    source = f"""def loop(pointer, output):
    while True:
{body}"""

    with pytest.raises(UnsupportedCapture, match="SOURCE_PROFILE") as error:
        profile_triton_source(source, {"pointer", "output"})

    assert "MEMORY_MODEL" not in str(error.value)


def test_pointer_roles_must_be_real_formals():
    source = """@triton.jit
def kernel(pointer):
    tl.load(pointer)
"""
    with pytest.raises(UnsupportedCapture, match="pointer argument roles"):
        profile_triton_source(source, {"not_a_formal"})


@pytest.mark.parametrize(
    "signature",
    ("source, destination=None", "source, *, destination", "source, /, destination"),
)
def test_kernel_parameters_must_be_required_and_positional(signature):
    source = f"""def kernel({signature}):
    tl.store(destination, tl.load(source))
"""
    with pytest.raises(UnsupportedCapture, match="required positional"):
        profile_triton_source(source, {"source", "destination"})


@pytest.mark.parametrize(
    ("source", "message"),
    (
        (
            """def kernel(tl, source, destination):
    tl.store(destination, tl.load(source))
""",
            "Triton namespace is shadowed",
        ),
        (
            """def kernel(source, destination, min):
    value = min(tl.load(source), 1.0)
    tl.store(destination, value)
""",
            "callable min is shadowed",
        ),
        (
            """def kernel(source, destination):
    value = unknown_value.to(tl.float32)
    tl.store(destination, tl.load(source) + value)
""",
            ".to receiver is not a supported Triton expression",
        ),
        (
            """def kernel(source, destination):
    value = unknown_value
    converted = value.to(tl.float32)
    tl.store(destination, tl.load(source) + converted)
""",
            ".to receiver is not a supported Triton expression",
        ),
        (
            """def kernel(source, destination):
    tl.store(destination, tl.load(source))
    return
""",
            "Return is outside",
        ),
        (
            """def kernel(source, destination):
    def hidden():
        tl.store(destination, 0.0)
    tl.store(destination, tl.load(source))
""",
            "FunctionDef is outside",
        ),
    ),
)
def test_source_profile_rejects_lexically_unbound_or_post_effect_forms(source, message):
    with pytest.raises(UnsupportedCapture, match=message):
        profile_triton_source(source, {"source", "destination"})
