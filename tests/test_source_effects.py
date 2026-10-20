from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from daggertrace.effects import (
    profile_triton_source,
)
from daggertrace.errors import UnsupportedCapture
from daggertrace.replay import ReplayError
from daggertrace.replay_source import _source_effects


_ACCEPTED_SOURCE_CASES = (
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source)
    tl.store(destination, value)
""",
        {"source", "destination"},
        (("source", "destination"), ("source",), ("destination",)),
        id="direct",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source)
    value = value.to(tl.float32)
    tl.store(destination, value)
""",
        {"source", "destination"},
        (("source", "destination"), ("source",), ("destination",)),
        id="sequential-value-rebind",
    ),
    pytest.param(
        """def kernel(source, destination):
    value, index = tl.max(tl.load(source), axis=0, return_indices=True)
    tl.store(destination, value)
""",
        {"source", "destination"},
        (("source", "destination"), ("source",), ("destination",)),
        id="tuple-result-binding",
    ),
    pytest.param(
        """def kernel(source, destination):
    read_pointer = source
    write_pointer = destination
    value = tl.load(read_pointer)
    tl.store(write_pointer, value)
""",
        {"source", "destination"},
        (("source", "destination"), ("source",), ("destination",)),
        id="local-alias",
    ),
    pytest.param(
        """def kernel(source, destination):
    read_base = source
    read_pointer = read_base
    write_base = destination
    write_pointer = write_base
    value = tl.load(read_pointer)
    tl.store(write_pointer, value)
""",
        {"source", "destination"},
        (("source", "destination"), ("source",), ("destination",)),
        id="chained-alias",
    ),
    pytest.param(
        """def kernel(source, destination, mask):
    value = tl.load(source, mask=mask, other=0.0)
    tl.store(destination, value, mask=mask)
""",
        {"source", "destination"},
        (("source", "destination", "mask"), ("source",), ("destination",)),
        id="masked",
    ),
    pytest.param(
        """def kernel(source, destination):
    mask = 1
    value = tl.load(source, mask=mask, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        (("source", "destination"), ("source",), ("destination",)),
        id="local-non-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination, mask):
    value = tl.load(source, mask=None or mask, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        (("source", "destination", "mask"), ("source",), ("destination",)),
        id="short-circuit-non-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination, mask):
    value = tl.load(source, mask=(None, mask)[1], other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        (("source", "destination", "mask"), ("source",), ("destination",)),
        id="literal-subscript-non-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination, flag):
    value = tl.load(source, mask=(False, True)[flag], other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        (("source", "destination", "flag"), ("source",), ("destination",)),
        id="dynamic-literal-non-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source, mask=False and None, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        (("source", "destination"), ("source",), ("destination",)),
        id="short-circuit-false-mask",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source, mask=True or None, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        (("source", "destination"), ("source",), ("destination",)),
        id="short-circuit-true-mask",
    ),
    pytest.param(
        """def kernel(source, destination, row_mask):
    value = tl.load(source, mask=row_mask[:, None], other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        (("source", "destination", "row_mask"), ("source",), ("destination",)),
        id="tensor-slice-mask",
    ),
    pytest.param(
        """def kernel(source, destination, mode: tl.constexpr):
    if mode == _CASTING_MODE_LLAMA:
        value = tl.load(source)
    if mode == _CASTING_MODE_LLAMA:
        tl.store(destination, value)
""",
        {"source", "destination"},
        (("source", "destination", "mode"), ("source",), ("destination",)),
        id="liger-constexpr-module-comparator",
    ),
    pytest.param(
        """def kernel(source, destination, k: tl.constexpr):
    if k > 0:
        for slot in tl.static_range(0, k):
            tl.store(destination + slot, tl.load(source + slot))
""",
        {"source", "destination"},
        (("source", "destination", "k"), ("source",), ("destination",)),
        id="art-constexpr-static-range",
    ),
    pytest.param(
        """def kernel(source, destination, rows):
    row_start = tl.program_id(0)
    row_end = min(row_start + 1, rows)
    for row in range(row_start, row_end):
        read_pointer = source + row
        value = tl.load(read_pointer)
        tl.store(destination + row, value)
""",
        {"source", "destination"},
        (("source", "destination", "rows"), ("source",), ("destination",)),
        id="liger-runtime-range",
    ),
    pytest.param(
        """def kernel(source, destination, flag: tl.constexpr):
    if not (flag + 1 <= 0) and True:
        value = tl.load(source)
        tl.store(destination, value)
""",
        {"source", "destination"},
        (("source", "destination", "flag"), ("source",), ("destination",)),
        id="constexpr-unary-binary-bool-compare",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source, mask=None)
    tl.store(destination, value)
""",
        {"source", "destination"},
        (("source", "destination"), ("source",), ("destination",)),
        id="none-mask-without-other",
    ),
)


_REJECTED_SOURCE_CASES = (
    pytest.param(
        """def kernel(source, destination):
    read_pointer = later_pointer
    value = tl.load(read_pointer)
    later_pointer = source
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="forward-pointer-alias",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = later_value
    tl.store(destination, value)
    later_value = 0.0
""",
        {"source", "destination"},
        id="forward-value-alias",
    ),
    pytest.param(
        """def kernel(source, destination):
    read_pointer = later_pointer + source
    later_pointer = 0
    value = tl.load(read_pointer)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="forward-pointer-later-assignment",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = later_value + 1
    later_value = 0.0
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="forward-value-later-assignment",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = 0.0
    value = later_value
    tl.store(destination, value)
    later_value = 0.0
""",
        {"source", "destination"},
        id="rebound-forward-value-alias",
    ),
    pytest.param(
        """def kernel(source, destination, runtime):
    if runtime:
        value = tl.load(source)
        tl.store(destination, value)
""",
        {"source", "destination"},
        id="runtime-if",
    ),
    pytest.param(
        """def kernel(source, destination, runtime):
    flag = runtime
    if flag:
        value = tl.load(source)
        tl.store(destination, value)
""",
        {"source", "destination"},
        id="runtime-local-if",
    ),
    pytest.param(
        """def kernel(source, destination, runtime, flag: tl.constexpr):
    if flag:
        runtime = unknown_value
    tl.store(destination, runtime)
""",
        {"source", "destination"},
        id="escaped-formal-rebinding",
    ),
    pytest.param(
        """def kernel(source, destination):
    if _CASTING_MODE_LLAMA:
        value = tl.load(source)
        tl.store(destination, value)
""",
        {"source", "destination"},
        id="bare-module-constant-if",
    ),
    pytest.param(
        """def kernel(source, destination, mode: tl.constexpr):
    if tl.program_id(0):
        value = tl.load(source)
        tl.store(destination, value)
""",
        {"source", "destination"},
        id="call-if",
    ),
    pytest.param(
        """def kernel(source, destination, mode: tl.constexpr):
    if mode.value:
        value = tl.load(source)
        tl.store(destination, value)
""",
        {"source", "destination"},
        id="attribute-if",
    ),
    pytest.param(
        """def kernel(source, destination, mode: tl.constexpr):
    if mode[0]:
        value = tl.load(source)
        tl.store(destination, value)
""",
        {"source", "destination"},
        id="subscript-if",
    ),
    pytest.param(
        """def kernel(source, destination, flag: tl.constexpr):
    if flag:
        read_pointer = source
    value = tl.load(read_pointer)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="escaped-branch-alias",
    ),
    pytest.param(
        """def kernel(source, destination, flag: tl.constexpr):
    value = tl.load(source)
    if flag:
        value = unknown_value
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="escaped-branch-rebinding",
    ),
    pytest.param(
        """def kernel(source, destination, flag: tl.constexpr):
    if flag:
        value = tl.load(source)
    value = later_value
    if flag:
        tl.store(destination, value)
""",
        {"source", "destination"},
        id="stale-guarded-value",
    ),
    pytest.param(
        """def kernel(source, destination, flag: tl.constexpr):
    if flag:
        value = unknown_value
    else:
        value = 0.0
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="escaped-open-branch-value",
    ),
    pytest.param(
        """def kernel(source, destination):
    for slot in range(0, 1):
        read_pointer = source + slot
    value = tl.load(read_pointer)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="escaped-loop-alias",
    ),
    pytest.param(
        """def kernel(source, destination, outer: tl.constexpr, flag: tl.constexpr):
    if outer:
        if flag:
            value = tl.load(source)
    if flag:
        tl.store(destination, value)
""",
        {"source", "destination"},
        id="escaped-nested-branch-value",
    ),
    pytest.param(
        """def kernel(source, destination, flag: tl.constexpr):
    for slot in range(0, 1):
        if flag:
            value = tl.load(source + slot)
    if flag:
        tl.store(destination, value)
""",
        {"source", "destination"},
        id="escaped-loop-guarded-value",
    ),
    pytest.param(
        """def kernel(source, destination, count):
    for slot in tl.static_range(0, count):
        value = tl.load(source + slot)
        tl.store(destination + slot, value)
""",
        {"source", "destination"},
        id="static-range-runtime-formal",
    ),
    pytest.param(
        """def kernel(source, destination):
    for slot in range(source):
        value = tl.load(destination)
        tl.store(destination, value)
""",
        {"source", "destination"},
        id="range-pointer-formal",
    ),
    pytest.param(
        """def kernel(source, destination):
    for slot in range(tl.load(source), 1):
        tl.store(destination, 0.0)
""",
        {"source", "destination"},
        id="range-memory-control",
    ),
    pytest.param(
        """def kernel(source, destination, range):
    for slot in range(1):
        tl.store(destination, 0.0)
""",
        {"source", "destination"},
        id="range-shadowed",
    ),
    pytest.param(
        """def kernel(source, destination, min):
    for slot in range(min(0, 1)):
        tl.store(destination, 0.0)
""",
        {"source", "destination"},
        id="range-bound-shadowed",
    ),
    pytest.param(
        """def kernel(source, destination, flag: tl.constexpr):
    value = tl.load(source) if flag else 0.0
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="conditional-expression",
    ),
    pytest.param(
        """def kernel(source, destination):
    destination = source
    value = tl.load(source)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="formal-rebind",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load()
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-missing-operand",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source, 0.0, 1.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-extra-operands",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source, mask=unknown_mask)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-open-operand",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-without-mask",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source, mask=None, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination):
    mask = None
    value = tl.load(source, mask=mask, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-none-mask-alias",
    ),
    pytest.param(
        """def kernel(source, destination):
    mask = 1
    mask = None
    value = tl.load(source, mask=mask, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-rebound-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination):
    mask = None or None
    value = tl.load(source, mask=mask, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-bool-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination):
    mask = (None,)[0]
    value = tl.load(source, mask=mask, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-subscript-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination):
    mask = (None,)
    value = tl.load(source, mask=mask[0], other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-named-tuple-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination, flag: tl.constexpr):
    value = tl.load(source, mask=(None, True)[flag], other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-dynamic-literal-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination, flag: tl.constexpr):
    value = tl.load(source, mask=((None,), (True,))[flag][0], other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-nested-dynamic-literal-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination, flag):
    masks = [None, True]
    value = tl.load(source, mask=masks[flag], other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-dynamic-named-list-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination):
    masks = ((None,),)
    value = tl.load(source, mask=masks[0][0], other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-nested-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination):
    mask, value = (None, 0.0)
    value = tl.load(source, mask=mask, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-unpacked-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination, mask):
    value = tl.load(source, mask=None and mask, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-and-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source, mask=True and None, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-terminal-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination, flag):
    value = tl.load(source, mask=flag and None, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-unknown-and-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination, flag):
    value = tl.load(source, mask=flag or None, other=0.0)
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="load-other-with-unknown-or-none-mask",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source)
    tl.store(destination)
""",
        {"source", "destination"},
        id="store-missing-operand",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source)
    tl.store(destination, value, 0.0, 1.0)
""",
        {"source", "destination"},
        id="store-extra-operands",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source)
    tl.store(destination, unknown_global)
""",
        {"source", "destination"},
        id="store-open-operand",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source)
    tl.store(destination, value, other=0.0)
""",
        {"source", "destination"},
        id="store-unexpected-keyword",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source)
    options = {}
    tl.store(destination, value, **options)
""",
        {"source", "destination"},
        id="starred-keyword",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source)
    value = other = unknown
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="chained-assignment",
    ),
    pytest.param(
        """def kernel(source, destination):
    value = tl.load(source)
    value[0] += unknown
    tl.store(destination, value)
""",
        {"source", "destination"},
        id="subscript-augmented-assignment",
    ),
    pytest.param(
        """def kernel(source, destination):
    marker: tl.load(source)
    tl.store(destination, 0.0)
""",
        {"source", "destination"},
        id="annotation-without-value",
    ),
)


@pytest.mark.parametrize("source,pointer_roles,expected", _ACCEPTED_SOURCE_CASES)
def test_source_effect_corpus_accepts_in_independent_v2_boundaries(
    source, pointer_roles, expected
):
    profile = profile_triton_source(source, pointer_roles)
    replay = _source_effects(source, pointer_roles)

    assert (profile.arguments, profile.reads, profile.writes) == expected
    assert replay == expected


@pytest.mark.parametrize(
    "bound",
    (
        pytest.param("(pointer,)[0]", id="tuple"),
        pytest.param("[pointer][0]", id="list"),
        pytest.param("([pointer][0:1])[0]", id="slice"),
        pytest.param("pointer or 1", id="bool"),
        pytest.param("pointer == 0", id="compare"),
        pytest.param("min(pointer, 1)", id="call-argument"),
        pytest.param("tl.max(0, axis=pointer)", id="call-keyword"),
        pytest.param("pointer.shape", id="attribute"),
        pytest.param("pointer.to(tl.int32)", id="to-receiver"),
    ),
)
def test_source_effect_corpus_rejects_pointer_wrapped_range_bounds_in_all_boundaries(bound):
    source = f"""def kernel(source, destination):
    pointer = source
    for slot in range({bound}):
        value = tl.load(source)
        tl.store(destination, value)
"""
    pointer_roles = {"source", "destination"}

    with pytest.raises(UnsupportedCapture):
        profile_triton_source(source, pointer_roles)
    with pytest.raises(ReplayError):
        _source_effects(source, pointer_roles)


@pytest.mark.parametrize(
    "bound",
    (
        pytest.param("(1,)[0]", id="tuple"),
        pytest.param("[1][0]", id="list"),
        pytest.param("([1][0:1])[0]", id="slice"),
        pytest.param("False or 1", id="bool"),
        pytest.param("1 == 1", id="compare"),
        pytest.param("min(1, 2)", id="call-argument"),
        pytest.param("tl.max(0, axis=1)", id="call-keyword"),
        pytest.param("(1).real", id="attribute"),
        pytest.param("tl.program_id(0).to(tl.int32)", id="to-receiver"),
    ),
)
def test_source_effect_corpus_accepts_scalar_wrapped_range_bounds_in_all_boundaries(bound):
    source = f"""def kernel(source, destination):
    for slot in range({bound}):
        value = tl.load(source)
        tl.store(destination, value)
"""
    pointer_roles = {"source", "destination"}
    expected = (("source", "destination"), ("source",), ("destination",))

    profile = profile_triton_source(source, pointer_roles)
    replay = _source_effects(source, pointer_roles)
    assert (profile.arguments, profile.reads, profile.writes) == expected
    assert replay == expected


def _assert_source_effects_rejected_in_all_boundaries(source):
    pointer_roles = {"source", "destination"}

    with pytest.raises(UnsupportedCapture):
        profile_triton_source(source, pointer_roles)
    with pytest.raises(ReplayError):
        _source_effects(source, pointer_roles)


def _assert_source_effects_accepted_in_all_boundaries(source):
    pointer_roles = {"source", "destination"}
    function = ast.parse(source).body[0]
    expected = (
        tuple(argument.arg for argument in function.args.args),
        ("source",),
        ("destination",),
    )

    profile = profile_triton_source(source, pointer_roles)
    replay = _source_effects(source, pointer_roles)

    assert (profile.arguments, profile.reads, profile.writes) == expected
    assert replay == expected


@pytest.mark.parametrize(
    "source",
    (
        pytest.param(
            """def kernel(source, destination):
    value = unknown_value
    for slot in range(0):
        value = 0.0
    tl.store(destination, value)
""",
            id="zero-trip-retains-open-value",
        ),
        pytest.param(
            """def kernel(source, destination):
    pointer = source
    for slot in range(2):
        value = tl.load(pointer)
        pointer = destination
        tl.store(destination, value)
""",
            id="later-pointer-root-transition",
        ),
        pytest.param(
            """def kernel(source, destination):
    value = 0.0
    for slot in range(2):
        tl.store(destination, value)
        value = unknown_value
""",
            id="later-value-open-transition",
        ),
        pytest.param(
            """def kernel(source, destination):
    mask = True
    for slot in range(2):
        value = tl.load(source, mask=mask, other=0.0)
        mask = None
    tl.store(destination, 0.0)
""",
            id="later-none-mask-transition",
        ),
        pytest.param(
            """def kernel(source, destination, flag: tl.constexpr):
    if flag:
        value = tl.load(source)
    for slot in range(2):
        if flag:
            tl.store(destination, value)
        value = unknown_value
""",
            id="later-guarded-value-transition",
        ),
        pytest.param(
            """def kernel(source, destination):
    flag = 1
    for slot in range(2):
        if flag:
            tl.store(destination, 0.0)
        flag = unknown_value
""",
            id="later-known-predicate-transition",
        ),
        pytest.param(
            """def kernel(source, destination, flag: tl.constexpr):
    if flag:
        index = None
    for index in range(1):
        value = tl.load(source)
    tl.store(destination, 0.0)
""",
            id="guarded-loop-target-transition",
        ),
    ),
)
def test_source_effects_reject_loop_carried_transitions_in_all_boundaries(source):
    _assert_source_effects_rejected_in_all_boundaries(source)


@pytest.mark.parametrize(
    "binding",
    (
        pytest.param("pointer = source.to(tl.int32)", id="to"),
        pytest.param("pointer = (source,)[0]", id="tuple-subscript"),
        pytest.param("pointer = [source][0]", id="list-subscript"),
        pytest.param("pointer, = (source,)", id="tuple-destructure"),
        pytest.param("[pointer] = [source]", id="list-destructure"),
        pytest.param("alias = source\n    pointer = alias", id="alias-chain"),
        pytest.param("pointer = source\n    pointer += 0", id="augmented-assignment"),
    ),
)
def test_source_effects_reject_hidden_pointer_range_bounds_in_all_boundaries(binding):
    source = f"""def kernel(source, destination):
    {binding}
    for slot in range(pointer):
        value = tl.load(source)
        tl.store(destination, value)
"""

    _assert_source_effects_rejected_in_all_boundaries(source)


def test_source_effects_reject_augmented_pointer_memory_use_in_all_boundaries():
    _assert_source_effects_rejected_in_all_boundaries(
        """def kernel(source, destination):
    pointer = source
    pointer += 0
    value = tl.load(pointer)
    tl.store(destination, value)
"""
    )


def test_source_effects_project_destructured_values_in_all_boundaries():
    _assert_source_effects_accepted_in_all_boundaries(
        """def kernel(source, destination):
    mask, unused = (True, None)
    value = tl.load(source, mask=mask, other=0.0)
    tl.store(destination, value)
"""
    )


def test_source_effects_ignore_min_key_as_a_result_candidate_in_all_boundaries():
    _assert_source_effects_accepted_in_all_boundaries(
        """def kernel(source, destination):
    mask = min(True, True, key=None)
    value = tl.load(source, mask=mask, other=0.0)
    tl.store(destination, value)
"""
    )


def test_source_effects_accept_exact_non_pointer_tuple_range_bound_in_all_boundaries():
    _assert_source_effects_accepted_in_all_boundaries(
        """def kernel(source, destination):
    for slot in range((0, source)[0]):
        value = tl.load(source)
        tl.store(destination, value)
"""
    )


@pytest.mark.parametrize(
    "source",
    (
        pytest.param(
            """def kernel(source, destination, flag):
    masks = (None,)
    masks += (True,)
    value = tl.load(source, mask=masks[flag], other=0.0)
    tl.store(destination, value)
""",
            id="augmented-container",
        ),
        pytest.param(
            """def kernel(source, destination, flag):
    masks = (None,)
    masks = masks + (True,)
    value = tl.load(source, mask=masks[flag], other=0.0)
    tl.store(destination, value)
""",
            id="concatenated-container",
        ),
        pytest.param(
            """def kernel(source, destination, flag):
    fallback = None
    mask = flag or fallback
    value = tl.load(source, mask=mask, other=0.0)
    tl.store(destination, value)
""",
            id="named-or",
        ),
        pytest.param(
            """def kernel(source, destination):
    value = tl.load(source, mask=min((None,)), other=0.0)
    tl.store(destination, value)
""",
            id="min-none",
        ),
        pytest.param(
            """def kernel(source, destination):
    value = tl.load(source, mask=1 + None, other=0.0)
    tl.store(destination, value)
""",
            id="binary-none",
        ),
    ),
)
def test_source_effects_reject_possible_none_masks_in_all_boundaries(source):
    _assert_source_effects_rejected_in_all_boundaries(source)


def test_source_effects_accepts_non_none_dynamic_literal_or_mask_in_all_boundaries():
    _assert_source_effects_accepted_in_all_boundaries(
        """def kernel(source, destination, flag):
    value = tl.load(source, mask=(None, True)[flag] or True, other=0.0)
    tl.store(destination, value)
"""
    )


@pytest.mark.parametrize(
    ("source", "rejected"),
    (
        pytest.param(
            """def kernel(source, destination, flag: tl.constexpr):
    if flag:
        value = tl.load(source)
    flag = 1
    if flag:
        tl.store(destination, value)
""",
            True,
            id="predicate-rebound",
        ),
        pytest.param(
            """def kernel(source, destination, flag: tl.constexpr):
    if flag:
        value = tl.load(source)
        if flag:
            flag = 1
    if flag:
        tl.store(destination, value)
""",
            True,
            id="nested-predicate-rebound",
        ),
        pytest.param(
            """def kernel(source, destination, flag: tl.constexpr):
    if flag:
        value = tl.load(source)
    unrelated = 1
    if flag:
        tl.store(destination, value)
""",
            False,
            id="unrelated-rebound",
        ),
        pytest.param(
            """def kernel(source, destination, flag: tl.constexpr):
    if flag:
        value = tl.load(source)
    if flag:
        tl.store(destination, value)
""",
            False,
            id="unchanged-predicate",
        ),
    ),
)
def test_source_effects_track_repeated_guard_predicate_bindings(source, rejected):
    if rejected:
        _assert_source_effects_rejected_in_all_boundaries(source)
    else:
        _assert_source_effects_accepted_in_all_boundaries(source)


@pytest.mark.parametrize(
    "capture_path",
    (
        pytest.param(
            "evidence/results/bundle/liger/liger-rmsnorm/candidate.capture.json",
            id="liger",
        ),
        pytest.param(
            "evidence/results/bundle/art/openpipe-art-local-stats/candidate.capture.json",
            id="art",
        ),
    ),
)
def test_checked_liger_and_art_sources_accept_in_independent_boundaries(capture_path):
    capture = json.loads(
        (Path(__file__).resolve().parents[1] / capture_path).read_text(encoding="utf-8")
    )
    for kernel in capture["kernels"]:
        pointer_roles = set(kernel["reads"]) | set(kernel["writes"])
        expected = (
            tuple(kernel["argument_names"]),
            tuple(kernel["source_reads"]),
            tuple(kernel["source_writes"]),
        )
        profile = profile_triton_source(kernel["source"], pointer_roles)
        replay = _source_effects(kernel["source"], pointer_roles)
        assert (profile.arguments, profile.reads, profile.writes) == expected
        assert replay == expected


@pytest.mark.parametrize("source,pointer_roles", _REJECTED_SOURCE_CASES)
def test_source_effect_corpus_rejects_in_independent_v2_boundaries(
    source, pointer_roles
):
    with pytest.raises(UnsupportedCapture):
        profile_triton_source(source, pointer_roles)
    with pytest.raises(ReplayError):
        _source_effects(source, pointer_roles)


@pytest.mark.skipif(not hasattr(ast, "TryStar"), reason="TryStar is unavailable")
def test_source_effect_corpus_rejects_trystar_in_v2_boundary():
    source = """def kernel(source, destination):
    try:
        value = tl.load(source)
    except* Exception:
        value = 0.0
    tl.store(destination, value)
"""
    pointer_roles = {"source", "destination"}

    with pytest.raises(UnsupportedCapture):
        profile_triton_source(source, pointer_roles)
    with pytest.raises(ReplayError):
        _source_effects(source, pointer_roles)


@pytest.mark.parametrize(
    ("source", "reason"),
    (
        pytest.param(
            """def kernel(source, destination):
    while True:
        tl.atomic_cas(source, 0, 1)
        break
    tl.store(destination, 0.0)
""",
            "MEMORY_MODEL",
            id="protocol-before-loop-control",
        ),
        pytest.param(
            """def kernel(source, destination):
    while True:
        value = tl.load(source)
        break
    tl.store(destination, value)
""",
            "SOURCE_PROFILE",
            id="non-protocol-loop-control",
        ),
        pytest.param(
            """def kernel(source, destination):
    tl = source
    tl.atomic_cas(destination, 0, 1)
""",
            "SOURCE_PROFILE",
            id="namespace-shadow-precedes-protocol",
        ),
    ),
)
def test_source_effect_corpus_classifies_protocols_before_control(source, reason):
    pointer_roles = {"source", "destination"}

    with pytest.raises(UnsupportedCapture) as producer:
        profile_triton_source(source, pointer_roles)
    with pytest.raises(ReplayError) as replay:
        _source_effects(source, pointer_roles)
    assert reason in str(producer.value)
    assert reason in str(replay.value)
