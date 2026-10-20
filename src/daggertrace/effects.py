from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass

from .errors import UnsupportedCapture
from .triton_protocols import (
    MEMORY_MODEL_PROTOCOL_DETAIL,
    MEMORY_MODEL_REASON,
    TRITON_PROTOCOL_OPERATIONS,
    _TRITON_SOURCE_BARE_CALLS,
    _TRITON_SOURCE_TL_OPERATIONS,
)


@dataclass(frozen=True, slots=True)
class SourceEffect:
    kind: str
    role: str
    line: int
    operation: str


@dataclass(frozen=True, slots=True)
class SourceEffects:
    arguments: tuple[str, ...]
    events: tuple[SourceEffect, ...]

    @property
    def reads(self) -> tuple[str, ...]:
        return tuple(sorted({effect.role for effect in self.events if effect.kind in {"read", "read_write"}}))

    @property
    def writes(self) -> tuple[str, ...]:
        return tuple(sorted({effect.role for effect in self.events if effect.kind in {"write", "read_write"}}))


_READ_CALLS = {"load"}
_WRITE_CALLS = {"store"}
_PURE_CALLS = _TRITON_SOURCE_TL_OPERATIONS - _READ_CALLS - _WRITE_CALLS
_ALLOWED_CALLS = _READ_CALLS | _WRITE_CALLS | _PURE_CALLS
_ALLOWED_BARE_CALLS = _TRITON_SOURCE_BARE_CALLS
_MEMORY_MODEL_CALLS = frozenset(TRITON_PROTOCOL_OPERATIONS)


def profile_triton_source(source: str, pointer_arguments: set[str] | None = None) -> SourceEffects:
    """Extract formal pointer roles from a closed, source-level Triton effect subset.

    This parser is deliberately independent of PyTorch's TTIR mutation analysis. It
    follows simple local pointer aliases and rejects any effect whose base formal is
    ambiguous instead of guessing from variable names.
    """

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError as exc:
        raise UnsupportedCapture(f"kernel source is not parseable Python: {exc}") from exc
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) != 1 or isinstance(functions[0], ast.AsyncFunctionDef):
        raise UnsupportedCapture("kernel source must contain exactly one synchronous function")
    function = functions[0]
    if (
        function.args.posonlyargs
        or function.args.kwonlyargs
        or function.args.defaults
        or any(value is not None for value in function.args.kw_defaults)
        or function.args.vararg
        or function.args.kwarg
    ):
        raise UnsupportedCapture(
            "only required positional Triton kernel parameters are inside the source profile"
        )
    arguments = tuple(arg.arg for arg in function.args.args)
    pointer_formals = set(arguments) if pointer_arguments is None else set(pointer_arguments)
    if not pointer_formals or not pointer_formals.issubset(arguments):
        raise UnsupportedCapture("pointer argument roles do not match the kernel signature")
    local_bindings = _local_bindings(function)
    if "tl" in local_bindings:
        raise UnsupportedCapture("SOURCE_PROFILE: the Triton namespace is shadowed locally")
    while_nodes = [node for node in ast.walk(function) if isinstance(node, ast.While)]
    memory_protocol_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _tl_operation(node.func) in _MEMORY_MODEL_CALLS
    ]
    if memory_protocol_calls:
        if any(
            protocol in set(ast.walk(loop))
            for loop in while_nodes
            for protocol in memory_protocol_calls
        ):
            raise UnsupportedCapture(
                f"{MEMORY_MODEL_REASON}: {MEMORY_MODEL_PROTOCOL_DETAIL}"
            )
        protocol = memory_protocol_calls[0]
        operation = _tl_operation(protocol.func)
        raise UnsupportedCapture(
            f"{MEMORY_MODEL_REASON}: Triton operation tl.{operation} requires a "
            f"thread-memory proof at line {protocol.lineno}"
        )
    unsupported_control = next(
        (
            node
            for node in ast.walk(function)
            if isinstance(
                node,
                (
                    ast.Assert,
                    ast.AsyncFor,
                    ast.AsyncFunctionDef,
                    ast.AsyncWith,
                    ast.Await,
                    ast.Break,
                    ast.ClassDef,
                    ast.Continue,
                    ast.Delete,
                    ast.FunctionDef,
                    ast.Global,
                    ast.Import,
                    ast.ImportFrom,
                    ast.Lambda,
                    ast.ListComp,
                    ast.DictComp,
                    ast.Match,
                    ast.Nonlocal,
                    ast.SetComp,
                    ast.GeneratorExp,
                    ast.Raise,
                    ast.Return,
                    ast.Try,
                    *((ast.TryStar,) if hasattr(ast, "TryStar") else ()),
                    ast.With,
                    ast.Yield,
                    ast.YieldFrom,
                ),
            )
            and node is not function
        ),
        None,
    )
    if unsupported_control is not None:
        raise UnsupportedCapture(
            "SOURCE_PROFILE: "
            f"{type(unsupported_control).__name__} is outside the supported source subset"
        )
    if while_nodes:
        raise UnsupportedCapture(
            "SOURCE_PROFILE: while-loop control flow is outside the supported source subset"
        )
    rebound_formals = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id in pointer_formals
    }
    if rebound_formals:
        raise UnsupportedCapture(
            "SOURCE_PROFILE: pointer formals cannot be rebound: "
            f"{sorted(rebound_formals)}"
        )
    effects = _preflight_source_paths(function, pointer_formals, local_bindings)
    if not effects:
        raise UnsupportedCapture("kernel source contains no supported Triton memory effects")
    effects.sort(key=lambda effect: (effect.line, effect.operation, effect.role))
    return SourceEffects(arguments=arguments, events=tuple(effects))


def _preflight_source_paths(
    function: ast.FunctionDef,
    pointer_formals: set[str],
    local_bindings: set[str],
) -> list[SourceEffect]:
    """Check source-order bindings on each admitted control-flow path.

    The later effect summary is deliberately not recovered from a whole-function
    assignment map: that would make a definition visible before its statement or
    outside the branch/loop that introduced a pointer alias.
    """

    formals = {argument.arg for argument in function.args.args}
    constexpr = {
        argument.arg
        for argument in function.args.args
        if isinstance(argument.annotation, ast.Attribute)
        and isinstance(argument.annotation.value, ast.Name)
        and argument.annotation.value.id == "tl"
        and argument.annotation.attr == "constexpr"
    }
    effects: list[SourceEffect] = []
    visited_calls: set[int] = set()
    # Snapshot each binding at its definition so a later statement cannot close it.
    opaque, open_value = object(), object()

    def closed(node: ast.AST, values: dict[str, object]) -> bool:
        if isinstance(node, ast.Name):
            if node.id in values:
                value = values[node.id]
                return value is not open_value
            return node.id in formals
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Attribute):
            return (
                isinstance(node.value, ast.Name) and node.value.id == "tl"
            ) or closed(node.value, values)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
                admitted = closed(node.func.value, values)
            else:
                operation = _tl_operation(node.func)
                admitted = operation in _ALLOWED_CALLS or (
                    isinstance(node.func, ast.Name)
                    and node.func.id in _ALLOWED_BARE_CALLS
                    and node.func.id not in local_bindings
                )
            return admitted and all(
                closed(value, values)
                for value in (*node.args, *(keyword.value for keyword in node.keywords))
            )
        if isinstance(node, ast.BinOp):
            return closed(node.left, values) and closed(node.right, values)
        if isinstance(node, ast.UnaryOp):
            return closed(node.operand, values)
        if isinstance(node, ast.BoolOp):
            return all(closed(value, values) for value in node.values)
        if isinstance(node, ast.Compare):
            return closed(node.left, values) and all(
                closed(value, values) for value in node.comparators
            )
        if isinstance(node, ast.Subscript):
            return closed(node.value, values) and closed(node.slice, values)
        if isinstance(node, ast.Slice):
            return all(
                bound is None or closed(bound, values)
                for bound in (node.lower, node.upper, node.step)
            )
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(closed(value, values) for value in node.elts)
        return False

    def literal_index(value: ast.AST) -> int | None:
        if isinstance(value, ast.Constant) and type(value.value) in (bool, int):
            return int(value.value)
        if (
            isinstance(value, ast.UnaryOp)
            and isinstance(value.op, (ast.UAdd, ast.USub))
            and isinstance(value.operand, ast.Constant)
            and type(value.operand.value) is int
        ):
            return value.operand.value if isinstance(value.op, ast.UAdd) else -value.operand.value
        return None

    def may_resolve_to_none(node: ast.AST, values: dict[str, object]) -> bool:
        def literal_truth(value: ast.AST) -> bool | None:
            if isinstance(value, ast.Constant):
                return bool(value.value)
            return None

        if isinstance(node, ast.Constant):
            return node.value is None
        if isinstance(node, ast.Name):
            return values.get(node.id, opaque) is None
        if isinstance(node, ast.UnaryOp):
            return may_resolve_to_none(node.operand, values)
        if isinstance(node, ast.BinOp):
            return may_resolve_to_none(node.left, values) or may_resolve_to_none(
                node.right, values
            )
        if isinstance(node, ast.BoolOp):
            # An opaque prefix may reach a later literal None.
            operands = node.values
            if isinstance(node.op, ast.And):
                for operand in operands:
                    if may_resolve_to_none(operand, values):
                        return True
                    if literal_truth(operand) is False:
                        return False
            elif isinstance(node.op, ast.Or):
                for index, operand in enumerate(operands):
                    if may_resolve_to_none(operand, values):
                        if index == len(operands) - 1:
                            return True
                        continue
                    if literal_truth(operand) is True:
                        return False
            return False
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "min"
        ):
            return any(
                compound_may_resolve_to_none(value, values)
                for value in node.args
            ) or any(
                keyword.arg == "default"
                and compound_may_resolve_to_none(keyword.value, values)
                for keyword in node.keywords
            )
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Subscript) and may_resolve_to_none(node.value, values):
                return True
            if isinstance(node.value, ast.Name) and values.get(node.value.id, opaque) is None:
                return True
            if isinstance(node.value, (ast.Tuple, ast.List)):
                index = literal_index(node.slice)
                if index is not None and -len(node.value.elts) <= index < len(node.value.elts):
                    value = node.value.elts[index]
                    return (
                        compound_may_resolve_to_none(value, values)
                        if isinstance(value, (ast.Tuple, ast.List))
                        else may_resolve_to_none(value, values)
                    )
                return any(
                    compound_may_resolve_to_none(value, values) for value in node.value.elts
                )
        return False

    def compound_may_resolve_to_none(node: ast.AST, values: dict[str, object]) -> bool:
        if isinstance(node, (ast.Tuple, ast.List)):
            return any(compound_may_resolve_to_none(value, values) for value in node.elts)
        return may_resolve_to_none(node, values)

    def roots(node: ast.AST, aliases: dict[str, frozenset[str]]) -> frozenset[str]:
        if isinstance(node, ast.Name):
            if node.id in pointer_formals:
                return frozenset((node.id,))
            return aliases.get(node.id, frozenset())
        if isinstance(node, ast.BinOp):
            return roots(node.left, aliases) | roots(node.right, aliases)
        if isinstance(node, ast.UnaryOp):
            return roots(node.operand, aliases)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to"
        ):
            return roots(node.func.value, aliases)
        if isinstance(node, ast.Subscript) and isinstance(node.value, (ast.Tuple, ast.List)):
            index = literal_index(node.slice)
            if index is not None and -len(node.value.elts) <= index < len(node.value.elts):
                return roots(node.value.elts[index], aliases)
        return frozenset()

    def range_roots(node: ast.AST, aliases: dict[str, frozenset[str]]) -> frozenset[str]:
        if isinstance(node, ast.Name):
            return roots(node, aliases)
        if isinstance(node, ast.Attribute):
            return range_roots(node.value, aliases)
        if isinstance(node, ast.BinOp):
            return range_roots(node.left, aliases) | range_roots(node.right, aliases)
        if isinstance(node, ast.UnaryOp):
            return range_roots(node.operand, aliases)
        if isinstance(node, ast.BoolOp):
            return frozenset().union(
                *(range_roots(value, aliases) for value in node.values)
            )
        if isinstance(node, ast.Compare):
            return range_roots(node.left, aliases) | frozenset().union(
                *(range_roots(value, aliases) for value in node.comparators)
            )
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, (ast.Tuple, ast.List)):
                index = literal_index(node.slice)
                if index is not None and -len(node.value.elts) <= index < len(node.value.elts):
                    return range_roots(node.value.elts[index], aliases)
                return frozenset().union(
                    *(range_roots(value, aliases) for value in node.value.elts)
                )
            return range_roots(node.value, aliases) | range_roots(node.slice, aliases)
        if isinstance(node, ast.Slice):
            return frozenset().union(
                *(
                    range_roots(bound, aliases)
                    for bound in (node.lower, node.upper, node.step)
                    if bound is not None
                )
            )
        if isinstance(node, (ast.Tuple, ast.List)):
            return frozenset().union(
                *(range_roots(value, aliases) for value in node.elts)
            )
        if isinstance(node, ast.Call):
            if _tl_operation(node.func) in _READ_CALLS | _WRITE_CALLS:
                return frozenset()
            receiver = (
                range_roots(node.func.value, aliases)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "to"
                else frozenset()
            )
            # Only an admitted .to receiver is an operand; function names are syntax.
            return receiver | frozenset().union(
                *(range_roots(value, aliases) for value in node.args),
                *(range_roots(keyword.value, aliases) for keyword in node.keywords),
            )
        return frozenset()

    def memory_value(node: ast.AST, values: dict[str, object]) -> bool:
        if isinstance(node, ast.IfExp):
            return False
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id] is not open_value
        if isinstance(node, ast.Call) and _tl_operation(node.func) in _READ_CALLS | _WRITE_CALLS:
            return False
        return closed(node, values)

    def condition(node: ast.AST, known: set[str]) -> tuple[bool, bool, bool]:
        """Return valid, constexpr-dependent, and module-constant flags."""

        if isinstance(node, ast.Constant):
            return True, False, False
        if isinstance(node, ast.Name):
            if node.id in known:
                return True, True, False
            if node.id in formals or node.id in local_bindings:
                return False, False, False
            return True, False, True
        if isinstance(node, ast.UnaryOp):
            valid, dependent, module = condition(node.operand, known)
            return valid and not module, dependent, module
        if isinstance(node, ast.BinOp):
            left = condition(node.left, known)
            right = condition(node.right, known)
            return left[0] and right[0] and not (left[2] or right[2]), left[1] or right[1], left[2] or right[2]
        if isinstance(node, ast.BoolOp):
            parts = [condition(value, known) for value in node.values]
            return all(part[0] and not part[2] for part in parts), any(part[1] for part in parts), any(part[2] for part in parts)
        if isinstance(node, ast.Compare):
            parts = [condition(node.left, known), *(condition(value, known) for value in node.comparators)]
            module_indexes = [index for index, part in enumerate(parts) if part[2]]
            if any(not part[0] for part in parts) or len(module_indexes) > 1:
                return False, False, False
            if not module_indexes:
                return True, any(part[1] for part in parts), False
            if len(parts) != 2:
                return False, False, False
            module_index = module_indexes[0]
            dependent = any(
                part[1] for index, part in enumerate(parts) if index != module_index
            )
            return dependent, dependent, True
        return False, False, False

    def static_scalar(node: ast.AST, known: set[str]) -> bool:
        valid, _, module = condition(node, known)
        return valid and not module

    def range_scalar(
        node: ast.AST,
        values: dict[str, object],
        aliases: dict[str, frozenset[str]],
    ) -> bool:
        return (
            not range_roots(node, aliases)
            and not any(
                isinstance(item, ast.Call)
                and _tl_operation(item.func) in _READ_CALLS | _WRITE_CALLS
                for item in ast.walk(node)
            )
            and closed(node, values)
        )

    def scoped_guards(
        guarded: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        return {key: dict(bindings) for key, bindings in guarded.items()}

    guard_names: dict[str, frozenset[str]] = {}

    def merge_guarded(
        guarded: dict[str, dict[str, object]],
        branches: tuple[dict[str, dict[str, object]], ...],
    ) -> None:
        for key in tuple(guarded):
            if any(key not in branch for branch in branches):
                guarded.pop(key)
                continue
            bindings = guarded[key]
            for name in tuple(bindings):
                states = tuple(branch.get(key, {}).get(name, open_value) for branch in branches)
                if any(state is open_value for state in states):
                    bindings.pop(name)
                elif any(state is None for state in states):
                    bindings[name] = None
                else:
                    bindings[name] = opaque

    def update_guarded(
        guarded: dict[str, dict[str, object]], names: tuple[str, ...], state: object
    ) -> None:
        bound = frozenset(names)
        for key in tuple(guarded):
            if guard_names.get(key, frozenset()) & bound:
                guarded.pop(key)
        for bindings in guarded.values():
            for name in names:
                if name in bindings:
                    if state is open_value:
                        bindings.pop(name)
                    else:
                        bindings[name] = state

    def merge_active_bindings(
        values: dict[str, object],
        aliases: dict[str, frozenset[str]],
        known: set[str],
        initial_values: dict[str, object],
        initial_aliases: dict[str, frozenset[str]],
        branches: tuple[
            tuple[dict[str, object], dict[str, frozenset[str]], set[str]], ...
        ],
    ) -> None:
        # A nested block can change only its pre-existing bindings when every
        # path retains a closed active value; new bindings stay lexical.
        for name, previous in initial_values.items():
            if any(branch_values[name] != previous for branch_values, _, _ in branches):
                states = tuple(branch_values[name] for branch_values, _, _ in branches)
                if any(state is open_value for state in states):
                    values[name] = open_value
                elif any(state is None for state in states):
                    values[name] = None
                else:
                    values[name] = opaque
        for name, previous in initial_aliases.items():
            if any(branch_aliases.get(name) != previous for _, branch_aliases, _ in branches):
                aliases.pop(name, None)
        known.intersection_update(*(branch_known for _, _, branch_known in branches))

    def assign_names(target: ast.AST) -> tuple[str, ...] | None:
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, (ast.Tuple, ast.List)):
            names: list[str] = []
            for item in target.elts:
                item_names = assign_names(item)
                if item_names is None:
                    return None
                names.extend(item_names)
            return tuple(names) if names else None
        return None

    def assigned_values(target: ast.AST, value: ast.AST) -> tuple[tuple[str, ast.AST], ...] | None:
        if isinstance(target, ast.Name):
            return ((target.id, value),)
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            if len(target.elts) != len(value.elts):
                return None
            result: list[tuple[str, ast.AST]] = []
            for item, item_value in zip(target.elts, value.elts):
                assigned = assigned_values(item, item_value)
                if assigned is None:
                    return None
                result.extend(assigned)
            return tuple(result)
        return None

    def call(node: ast.Call, values: dict[str, object], aliases: dict[str, frozenset[str]]) -> None:
        if id(node) in visited_calls:
            return
        visited_calls.add(id(node))
        operation = _tl_operation(node.func)
        if operation is None:
            if isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_BARE_CALLS:
                if node.func.id in local_bindings:
                    raise UnsupportedCapture(
                        f"SOURCE_PROFILE: supported callable {node.func.id} is shadowed locally"
                    )
                return
            if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
                if not closed(node.func.value, values):
                    raise UnsupportedCapture(
                        f"SOURCE_PROFILE: .to receiver is not a supported Triton expression at line {node.lineno}"
                    )
                return
            raise UnsupportedCapture(
                f"call {_call_name(node.func)} is outside the supported source subset at line {node.lineno}"
            )
        if operation not in _ALLOWED_CALLS:
            raise UnsupportedCapture(
                f"Triton operation tl.{operation} is outside the supported source subset at line {node.lineno}"
            )
        if operation in _PURE_CALLS:
            return
        positional_count = 1 if operation in _READ_CALLS else 2
        allowed_keywords = {"mask", "other"} if operation in _READ_CALLS else {"mask"}
        if len(node.args) != positional_count:
            raise UnsupportedCapture(
                f"SOURCE_PROFILE: tl.{operation} requires exactly {positional_count} positional operands at line {node.lineno}"
            )
        keyword_names = [keyword.arg for keyword in node.keywords]
        if (
            any(name is None for name in keyword_names)
            or len(set(keyword_names)) != len(keyword_names)
            or not set(keyword_names).issubset(allowed_keywords)
        ):
            raise UnsupportedCapture(
                f"SOURCE_PROFILE: tl.{operation} has unsupported or duplicate keywords at line {node.lineno}"
            )
        keyword_values = {keyword.arg: keyword.value for keyword in node.keywords}
        if operation == "load" and "other" in keyword_values:
            mask = keyword_values.get("mask")
            if mask is None or may_resolve_to_none(mask, values):
                raise UnsupportedCapture(
                    f"SOURCE_PROFILE: tl.load other requires a non-None mask at line {node.lineno}"
                )
        operands = (*node.args, *(keyword.value for keyword in node.keywords))
        admitted_nested_load = (
            operation == "store"
            and isinstance(node.args[1], ast.Call)
            and _tl_operation(node.args[1].func) == "load"
        )
        if admitted_nested_load:
            call(node.args[1], values, aliases)
        if any(
            not memory_value(operand, values)
            for index, operand in enumerate(operands)
            if not (admitted_nested_load and index == 1)
        ):
            raise UnsupportedCapture(
                f"SOURCE_PROFILE: tl.{operation} has an open operand at line {node.lineno}"
            )
        pointer_roots = roots(node.args[0], aliases)
        if len(pointer_roots) != 1:
            raise UnsupportedCapture(
                f"tl.{operation} pointer has {len(pointer_roots)} formal roots at line {node.lineno}: {sorted(pointer_roots)}"
            )
        effects.append(
            SourceEffect(
                kind="read" if operation in _READ_CALLS else "write",
                role=next(iter(pointer_roots)),
                line=node.lineno,
                operation=f"tl.{operation}",
            )
        )

    def statements(
        body: list[ast.stmt],
        values: dict[str, object],
        aliases: dict[str, frozenset[str]],
        known: set[str],
        guarded: dict[str, dict[str, object]],
    ) -> tuple[dict[str, object], dict[str, frozenset[str]], set[str]]:
        for node in body:
            if isinstance(node, ast.If):
                valid, _, module = condition(node.test, known)
                if not valid or (module and not isinstance(node.test, ast.Compare)):
                    raise UnsupportedCapture(
                        f"SOURCE_PROFILE: if predicate is not compile-time closed at line {node.lineno}"
                    )
                key = ast.dump(node.test, include_attributes=False)
                guard_names[key] = frozenset(
                    item.id
                    for item in ast.walk(node.test)
                    if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                )
                initial_values = dict(values)
                initial_aliases = dict(aliases)
                body_guarded = scoped_guards(guarded)
                body_guarded.setdefault(key, {})
                body_values = dict(values)
                body_values.update(body_guarded[key])
                body_aliases = dict(aliases)
                body_known = set(known)
                body_values, body_aliases, body_known = statements(
                    node.body, body_values, body_aliases, body_known, body_guarded
                )
                else_guarded = scoped_guards(guarded)
                else_guarded.setdefault(key, {})
                else_values, else_aliases, else_known = statements(
                    node.orelse, dict(values), dict(aliases), set(known), else_guarded
                )
                merge_active_bindings(
                    values,
                    aliases,
                    known,
                    initial_values,
                    initial_aliases,
                    (
                        (body_values, body_aliases, body_known),
                        (else_values, else_aliases, else_known),
                    ),
                )
                merge_guarded(guarded, (body_guarded, else_guarded))
                body_added = set(body_values) - set(initial_values) - set(guarded.get(key, {}))
                else_added = set(else_values) - set(initial_values)
                if key in body_guarded and key in else_guarded:
                    for name in body_added - set(body_aliases):
                        if body_values[name] is not open_value:
                            guarded.setdefault(key, {})[name] = body_values[name]
                for name in (body_added & else_added) - set(body_aliases) - set(else_aliases):
                    if body_values[name] is open_value or else_values[name] is open_value:
                        values[name] = open_value
                    elif body_values[name] is None or else_values[name] is None:
                        values[name] = None
                    else:
                        values[name] = opaque
                    guarded.get(key, {}).pop(name, None)
                    if name in body_known and name in else_known:
                        known.add(name)
                continue
            if isinstance(node, ast.For):
                static_range = isinstance(node.iter, ast.Call) and _tl_operation(node.iter.func) == "static_range"
                if (
                    not isinstance(node.target, ast.Name)
                    or node.target.id in formals
                    or node.orelse
                    or not isinstance(node.iter, ast.Call)
                    or node.iter.keywords
                    or not (
                        isinstance(node.iter.func, ast.Name)
                        and node.iter.func.id == "range"
                        and node.iter.func.id not in local_bindings
                        or static_range
                    )
                    or not 1 <= len(node.iter.args) <= 3
                    or any(
                        not static_scalar(value, known)
                        if static_range
                        else not range_scalar(value, values, aliases)
                        for value in node.iter.args
                    )
                ):
                    raise UnsupportedCapture(
                        f"SOURCE_PROFILE: for-loop control flow is outside the supported source subset at line {node.lineno}"
                    )
                initial_values = dict(values)
                initial_aliases = dict(aliases)
                initial_known = set(known)
                initial_guarded = scoped_guards(guarded)
                loop_guarded = scoped_guards(guarded)
                update_guarded(loop_guarded, (node.target.id,), opaque)
                loop_values = dict(values)
                loop_values[node.target.id] = opaque
                loop_aliases = dict(aliases)
                loop_aliases.pop(node.target.id, None)
                loop_known = set(known)
                loop_known.discard(node.target.id)
                loop_values, loop_aliases, loop_known = statements(
                    node.body,
                    loop_values,
                    loop_aliases,
                    loop_known,
                    loop_guarded,
                )
                preexisting = set(initial_values) | set(initial_aliases)
                stable_values = all(
                    loop_values.get(name, open_value)
                    is initial_values.get(name, open_value)
                    for name in preexisting
                )
                stable_aliases = all(
                    (name in loop_aliases, loop_aliases.get(name))
                    == (name in initial_aliases, initial_aliases.get(name))
                    for name in preexisting
                )
                stable_known = all(
                    (name in loop_known) == (name in initial_known) for name in preexisting
                )
                stable_guarded = all(
                    key in loop_guarded
                    and all(
                        loop_guarded[key].get(name, open_value) is state
                        for name, state in bindings.items()
                    )
                    for key, bindings in initial_guarded.items()
                )
                if not (stable_values and stable_aliases and stable_known and stable_guarded):
                    raise UnsupportedCapture(
                        f"SOURCE_PROFILE: for-loop bindings are not stable at line {node.lineno}"
                    )
                continue
            assignment: tuple[tuple[str, ...], ast.AST] | None = None
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    raise UnsupportedCapture(
                        f"SOURCE_PROFILE: assignment is outside the supported source subset at line {node.lineno}"
                    )
                names = assign_names(node.targets[0])
                if names is None:
                    raise UnsupportedCapture(
                        f"SOURCE_PROFILE: assignment is outside the supported source subset at line {node.lineno}"
                    )
                assignment = (names, node.value)
            elif isinstance(node, ast.AnnAssign):
                if node.value is None:
                    raise UnsupportedCapture(
                        f"SOURCE_PROFILE: assignment is outside the supported source subset at line {node.lineno}"
                    )
                names = assign_names(node.target)
                if names is None:
                    raise UnsupportedCapture(
                        f"SOURCE_PROFILE: assignment is outside the supported source subset at line {node.lineno}"
                    )
                assignment = (names, node.value)
            elif isinstance(node, ast.AugAssign) and not isinstance(node.target, ast.Name):
                raise UnsupportedCapture(
                    f"SOURCE_PROFILE: assignment is outside the supported source subset at line {node.lineno}"
                )
            if any(isinstance(nested, (ast.IfExp, ast.NamedExpr)) for nested in ast.walk(node)):
                raise UnsupportedCapture(
                    f"SOURCE_PROFILE: assignment is outside the supported source subset at line {node.lineno}"
                )
            for nested in ast.walk(node):
                if isinstance(nested, ast.Call):
                    call(nested, values, aliases)
            if assignment is not None:
                names, value = assignment
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                bindings = assigned_values(target, value)
                if bindings is None:
                    bindings = tuple((name, value) for name in names)
                contains_memory = any(
                    isinstance(item, ast.Call) and _tl_operation(item.func) in _READ_CALLS | _WRITE_CALLS
                    for item in ast.walk(value)
                )
                value_closed = closed(value, values)
                if contains_memory and not value_closed:
                    raise UnsupportedCapture(
                        f"SOURCE_PROFILE: assignment has an open value at line {node.lineno}"
                    )
                for name, bound_value in bindings:
                    bound_closed = closed(bound_value, values)
                    pointer_roots = roots(bound_value, aliases)
                    state = (
                        None
                        if may_resolve_to_none(bound_value, values)
                        or (
                            isinstance(bound_value, (ast.Tuple, ast.List))
                            and compound_may_resolve_to_none(bound_value, values)
                        )
                        else opaque
                        if bound_closed
                        else open_value
                    )
                    values[name] = state
                    if pointer_roots and bound_closed:
                        aliases[name] = pointer_roots
                    else:
                        aliases.pop(name, None)
                    update_guarded(guarded, (name,), state)
                known.difference_update(names)
                constant_value = condition(value, known)
                if constant_value[0] and not constant_value[2]:
                    known.update(names)
                continue
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                if node.target.id not in values and node.target.id not in formals:
                    raise UnsupportedCapture(
                        f"SOURCE_PROFILE: assignment has an open value at line {node.lineno}"
                    )
                if not closed(node.value, values):
                    raise UnsupportedCapture(
                        f"SOURCE_PROFILE: assignment has an open value at line {node.lineno}"
                    )
                name = node.target.id
                pointer_tainted = (
                    values.get(name, opaque) is open_value
                    or bool(aliases.get(name))
                    or bool(range_roots(node.value, aliases))
                )
                state = (
                    open_value
                    if pointer_tainted
                    else None
                    if may_resolve_to_none(node.target, values)
                    or may_resolve_to_none(node.value, values)
                    else opaque
                )
                values[name] = state
                aliases.pop(name, None)
                update_guarded(guarded, (name,), state)
                known.discard(node.target.id)
        return values, aliases, known

    statements(function.body, dict.fromkeys(formals, opaque), {}, constexpr, {})
    return effects


def _assignments(function: ast.FunctionDef) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    ambiguous: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in result:
                ambiguous.add(name)
            else:
                result[name] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            name = node.target.id
            if name in result:
                ambiguous.add(name)
            else:
                result[name] = node.value
        elif isinstance(node, (ast.AugAssign, ast.NamedExpr)):
            target = node.target
            if isinstance(target, ast.Name):
                ambiguous.add(target.id)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name in result:
                ambiguous.add(name)
            else:
                result[name] = node.iter
    for name in ambiguous:
        result.pop(name, None)
    return result


def _local_bindings(function: ast.FunctionDef) -> set[str]:
    bindings = {argument.arg for argument in function.args.args}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bindings.add(node.id)
    return bindings


def _closed_local_bindings(function: ast.FunctionDef, arguments: set[str]) -> set[str]:
    definitions: dict[str, list[ast.AST]] = {}
    for node in ast.walk(function):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                for name in (
                    item.id
                    for item in ast.walk(target)
                    if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
                ):
                    definitions.setdefault(name, []).append(node.value)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            definitions.setdefault(node.target.id, []).append(node.iter)
        elif isinstance(node, (ast.AugAssign, ast.NamedExpr)) and isinstance(
            node.target, ast.Name
        ):
            definitions.setdefault(node.target.id, []).append(node.value)

    closed = set(arguments)
    changed = True
    while changed:
        changed = False
        for name, values in definitions.items():
            if name not in closed and any(
                _closed_value(value, closed, {}, set()) for value in values
            ):
                closed.add(name)
                changed = True

    changed = True
    while changed:
        changed = False
        for name in tuple(closed - arguments):
            values = definitions[name]
            has_grounded_definition = any(
                _closed_value(value, closed - {name}, {}, set()) for value in values
            )
            all_definitions_closed = all(
                _closed_value(value, closed, {}, set()) for value in values
            )
            if not has_grounded_definition or not all_definitions_closed:
                closed.remove(name)
                changed = True
    return closed - arguments


def _closed_value(
    node: ast.AST,
    arguments: set[str],
    assignments: dict[str, ast.AST],
    seen: set[str],
) -> bool:
    if isinstance(node, ast.Name):
        if node.id in arguments:
            return True
        if node.id in seen or node.id not in assignments:
            return False
        return _closed_value(assignments[node.id], arguments, assignments, seen | {node.id})
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "tl":
            return True
        return _closed_value(node.value, arguments, assignments, seen)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            return _closed_value(node.func.value, arguments, assignments, seen) and all(
                _closed_value(value, arguments, assignments, seen)
                for value in (*node.args, *(keyword.value for keyword in node.keywords))
            )
        admitted = _tl_operation(node.func) in _ALLOWED_CALLS or (
            isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_BARE_CALLS
        )
        return admitted and all(
            _closed_value(value, arguments, assignments, seen)
            for value in (*node.args, *(keyword.value for keyword in node.keywords))
        )
    if isinstance(node, ast.BinOp):
        return _closed_value(node.left, arguments, assignments, seen) and _closed_value(
            node.right, arguments, assignments, seen
        )
    if isinstance(node, ast.UnaryOp):
        return _closed_value(node.operand, arguments, assignments, seen)
    if isinstance(node, ast.BoolOp):
        return all(_closed_value(value, arguments, assignments, seen) for value in node.values)
    if isinstance(node, ast.Compare):
        return _closed_value(node.left, arguments, assignments, seen) and all(
            _closed_value(value, arguments, assignments, seen) for value in node.comparators
        )
    if isinstance(node, ast.IfExp):
        return all(
            _closed_value(value, arguments, assignments, seen)
            for value in (node.test, node.body, node.orelse)
        )
    if isinstance(node, ast.Subscript):
        return _closed_value(node.value, arguments, assignments, seen) and _closed_value(
            node.slice, arguments, assignments, seen
        )
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_closed_value(value, arguments, assignments, seen) for value in node.elts)
    return False


def _tl_operation(function: ast.AST) -> str | None:
    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name) and function.value.id == "tl":
        return function.attr
    return None


def _call_name(function: ast.AST) -> str:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return type(function).__name__
