"""Source slicing and effect validation for replay."""

from __future__ import annotations

import ast
import io
import textwrap
import tokenize
from collections.abc import Iterator

from .triton_protocols import (
    _TRITON_SOURCE_BARE_CALLS,
    _TRITON_SOURCE_TL_OPERATIONS,
    UNKNOWN_MEMORY_MODEL_DETAIL,
    triton_protocol_kind,
)

from .replay_packet import ReplayError


def _enforce_ast_bounds(tree: ast.AST) -> None:
    """Enforce the bounded source AST envelope used by replay."""

    queue = [(tree, 0)]
    statements = 0
    for node, depth in queue:
        if depth > 128:
            raise ValueError("target source AST exceeds its depth bound")
        if isinstance(node, ast.stmt):
            statements += 1
            if statements > 4096:
                raise ValueError("target source AST exceeds its statement bound")
        children = tuple(ast.iter_child_nodes(node))
        if len(queue) + len(children) > 16384:
            raise ValueError("target source AST exceeds its node bound")
        queue.extend((child, depth + 1) for child in children)


def _function_slice(source: str, name: str, *, bounded: bool = False) -> str:
    encoded = source.encode("utf-8")
    match_count = 0
    selected: tuple[int, int, bool] | None = None
    for candidate_name, start, raw_end, is_async in _source_function_fragments(source):
        if candidate_name != name:
            continue
        match_count += 1
        selected = (start, raw_end, is_async)
    if match_count != 1 or selected is None or selected[2]:
        raise ValueError(f"decorated top-level function {name!r} is not unique")
    start, raw_end, _ = selected
    fragment_encoded = encoded[start:raw_end]
    fragment = fragment_encoded.decode("utf-8")
    tree = ast.parse(fragment)
    if bounded:
        _enforce_ast_bounds(tree)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
        and node.decorator_list
    ]
    if len(functions) != 1 or functions[0].end_lineno is None:
        raise ValueError(f"decorated top-level function {name!r} is not unique")
    node = functions[0]
    fragment_lines = _source_line_starts(fragment_encoded)
    end = fragment_lines[node.end_lineno - 1] + node.end_col_offset
    if end > len(fragment_encoded):
        raise ValueError(f"function {name!r} has an invalid source extent")
    if encoded[start + end : start + end + 2] == b"\r\n":
        end += 2
    elif encoded[start + end : start + end + 1] in {b"\r", b"\n"}:
        end += 1
    return encoded[start : start + end].decode("utf-8")


def _source_function_fragments(source: str) -> Iterator[tuple[str, int, int, bool]]:
    """Yield decorated top-level function extents without parsing the whole file.

    The host AST sees only the selected function fragment.  Other syntax is checked
    lexically and structurally, not semantically parsed.  Tokenizable but semantically
    incomplete unbound suffixes such as `+`, `a=`, and `x=*`, as well as newer syntax,
    therefore remain opaque; this is what lets checked packets bind across Python
    versions.
    """

    encoded = source.encode("utf-8")
    lines: list[str] = []
    line_start = 0
    for index, char in enumerate(source):
        if char == "\n" or (char == "\r" and source[index + 1 : index + 2] != "\n"):
            lines.append(source[line_start : index + 1])
            line_start = index + 1
    if line_start < len(source):
        lines.append(source[line_start:])
    line_starts = _source_line_starts(encoded)
    # Replace lone CRs one-for-one; token columns must still index original bytes.
    token_source = "".join(
        "\n"
        if char == "\r" and (index + 1 == len(source) or source[index + 1] != "\n")
        else char
        for index, char in enumerate(source)
    )

    delimiter_closers = {"(": ")", "[": "]", "{": "}"}
    delimiter_stack: list[str] = []
    indent_stack: list[str] = []
    decorator_start: int | None = None
    decorator_expression = False
    awaiting_async = False
    function_name: str | None = None
    function_start: int | None = None
    function_async = False
    function_phase: str | None = None
    lambda_pending = 0

    def syntax_error(message: str, token: tokenize.TokenInfo) -> SyntaxError:
        return SyntaxError(message, ("<source>", token.start[0], token.start[1] + 1, ""))

    def finish(token: tokenize.TokenInfo) -> tuple[str, int, int, bool]:
        nonlocal function_name, function_start, function_phase
        if function_name is None or function_start is None:
            raise syntax_error("function scanner lost state", token)
        candidate = (
            function_name,
            function_start,
            _source_byte_offset(encoded, lines, line_starts, token.start),
            function_async,
        )
        function_name = None
        function_start = None
        function_phase = None
        return candidate

    try:
        for token in tokenize.generate_tokens(io.StringIO(token_source).readline):
            if token.type == tokenize.ERRORTOKEN:
                raise syntax_error(f"invalid token {token.string!r}", token)
            if token.type == tokenize.OP:
                if token.string not in tokenize.EXACT_TOKEN_TYPES:
                    raise syntax_error(f"invalid operator {token.string!r}", token)
                if token.string in delimiter_closers:
                    delimiter_stack.append(token.string)
                elif token.string in delimiter_closers.values():
                    if not delimiter_stack:
                        raise syntax_error(f"unmatched closing delimiter {token.string!r}", token)
                    opening = delimiter_stack.pop()
                    if delimiter_closers[opening] != token.string:
                        raise syntax_error(
                            f"mismatched delimiter {opening!r} and {token.string!r}", token
                        )
            if token.type == tokenize.INDENT:
                indent_stack.append(token.string)
            elif token.type == tokenize.DEDENT:
                if not indent_stack:
                    raise syntax_error("unmatched dedent", token)
                indent_stack.pop()

            if function_phase is not None:
                if function_phase == "awaiting_name":
                    if token.type != tokenize.NAME:
                        raise syntax_error("function definition has no name", token)
                    function_name = token.string
                    function_phase = "header"
                    lambda_pending = 0
                    continue
                if function_phase == "header":
                    if token.type == tokenize.NAME and token.string == "lambda" and not delimiter_stack:
                        lambda_pending += 1
                    elif token.type == tokenize.OP and token.string == ":" and not delimiter_stack:
                        if lambda_pending:
                            lambda_pending -= 1
                        else:
                            function_phase = "suite"
                    elif token.type == tokenize.ENDMARKER or (
                        token.type == tokenize.NEWLINE and not delimiter_stack
                    ):
                        raise syntax_error("function definition has no suite separator", token)
                    continue
                if function_phase == "suite":
                    if token.type == tokenize.COMMENT:
                        continue
                    if token.type == tokenize.NEWLINE:
                        function_phase = "indent"
                        continue
                    if token.type == tokenize.ENDMARKER:
                        raise syntax_error("function definition has no suite", token)
                    function_phase = "simple"
                if function_phase == "indent":
                    if token.type in {tokenize.COMMENT, tokenize.NL}:
                        continue
                    if token.type == tokenize.INDENT:
                        function_phase = "block"
                        continue
                    raise syntax_error("function definition has no indented suite", token)
                if function_phase == "block":
                    if token.type == tokenize.DEDENT and not indent_stack:
                        yield finish(token)
                        continue
                    if token.type == tokenize.ENDMARKER:
                        raise syntax_error("function definition has no terminating dedent", token)
                    continue
                if function_phase == "simple":
                    if token.type == tokenize.NEWLINE and not delimiter_stack:
                        yield finish(token)
                        continue
                    if token.type == tokenize.ENDMARKER:
                        yield finish(token)
                        return
                    continue
                raise RuntimeError(f"unknown function scan phase {function_phase!r}")

            if indent_stack or delimiter_stack:
                continue
            if token.type in {tokenize.INDENT, tokenize.DEDENT}:
                continue
            if token.type in {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING}:
                continue
            prefix = lines[token.start[0] - 1][: token.start[1]] if token.start[0] <= len(lines) else ""
            at_line_start = token.start[1] == 0 or all(char in "\t\f" for char in prefix)
            if decorator_start is None:
                if token.type == tokenize.OP and token.string == "@" and at_line_start:
                    decorator_start = _source_byte_offset(
                        encoded, lines, line_starts, (token.start[0], 0)
                    )
                    decorator_expression = True
                continue
            if decorator_expression:
                if token.type == tokenize.NEWLINE:
                    decorator_expression = False
                continue
            if token.type == tokenize.OP and token.string == "@" and at_line_start:
                decorator_expression = True
                awaiting_async = False
                continue
            if token.type == tokenize.NEWLINE:
                if awaiting_async:
                    raise syntax_error(
                        "async is not followed by a function definition", token
                    )
                continue
            if token.type == tokenize.NAME and token.string == "async":
                awaiting_async = True
                continue
            if token.type == tokenize.NAME and token.string == "def":
                function_name = None
                function_start = decorator_start
                function_async = awaiting_async
                function_phase = "awaiting_name"
                decorator_start = None
                decorator_expression = False
                awaiting_async = False
                continue
            if token.type == tokenize.ENDMARKER:
                continue
            decorator_start = None
            decorator_expression = False
            awaiting_async = False
    except tokenize.TokenError as exc:
        message, position = exc.args
        raise SyntaxError(message, ("<source>", position[0], position[1] + 1, "")) from exc

    if function_phase is not None:
        raise SyntaxError("unfinished function definition")
    if decorator_expression:
        raise SyntaxError("unfinished decorator expression")
    if awaiting_async:
        raise SyntaxError("unfinished async function definition")
    if decorator_start is not None:
        raise SyntaxError("unfinished decorated function")
    if delimiter_stack:
        raise SyntaxError("unclosed delimiter")
    if indent_stack:
        raise SyntaxError("unclosed indentation block")


def _source_byte_offset(
    encoded: bytes,
    lines: list[str],
    line_starts: list[int],
    position: tuple[int, int],
) -> int:
    line_number, column = position
    if line_number == len(lines) + 1 and column == 0:
        return len(encoded)
    if not 1 <= line_number <= len(lines) or not 0 <= column <= len(lines[line_number - 1]):
        raise ValueError("source token has an invalid position")
    return line_starts[line_number - 1] + len(lines[line_number - 1][:column].encode("utf-8"))


def _host_syntax_error(label: str, exc: SyntaxError) -> ReplayError:
    detail = exc.msg or str(exc)
    location = ""
    if exc.lineno is not None:
        location = f" at line {exc.lineno}"
        if exc.offset is not None:
            location += f", column {exc.offset}"
    return ReplayError(f"{label} could not be parsed by the host Python parser{location}: {detail}")


def _source_line_starts(encoded: bytes) -> list[int]:
    starts = [0]
    for index, byte in enumerate(encoded):
        if byte == 0x0A or (byte == 0x0D and encoded[index + 1 : index + 2] != b"\n"):
            starts.append(index + 1)
    return starts

def _source_effects(source: str, pointer_roles: set[str]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    tree = ast.parse(textwrap.dedent(source))
    try:
        _enforce_ast_bounds(tree)
    except ValueError as exc:
        raise ReplayError(str(exc)) from exc
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) != 1 or isinstance(functions[0], ast.AsyncFunctionDef):
        raise ReplayError("kernel source no longer has one function")
    function = functions[0]
    if (
        function.args.posonlyargs
        or function.args.kwonlyargs
        or function.args.defaults
        or any(value is not None for value in function.args.kw_defaults)
        or function.args.vararg
        or function.args.kwarg
    ):
        raise ReplayError(
            "SOURCE_PROFILE: only required positional Triton kernel parameters are supported"
        )
    argument_order = tuple(argument.arg for argument in function.args.args)
    arguments = set(argument_order)
    if not pointer_roles or not pointer_roles.issubset(arguments):
        raise ReplayError("HOP pointer roles do not match kernel formals")
    local_bindings = {argument.arg for argument in function.args.args}
    local_bindings.update(
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    if "tl" in local_bindings:
        raise ReplayError("SOURCE_PROFILE: the Triton namespace is shadowed locally")
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and triton_protocol_kind(f"tl.{node.func.attr}") is not None
        for node in ast.walk(function)
    ):
        raise ReplayError(UNKNOWN_MEMORY_MODEL_DETAIL)
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
        raise ReplayError(
            "SOURCE_PROFILE: "
            f"{type(unsupported_control).__name__} is outside the supported source subset"
        )
    while_nodes = [node for node in ast.walk(function) if isinstance(node, ast.While)]
    if while_nodes:
        raise ReplayError("SOURCE_PROFILE: while-loop control flow is outside the supported source subset")
    rebound_formals = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id in pointer_roles
    }
    if rebound_formals:
        raise ReplayError(
            "SOURCE_PROFILE: pointer formals cannot be rebound: "
            f"{sorted(rebound_formals)}"
        )
    return _source_path_effects(function, pointer_roles, local_bindings)


def _source_path_effects(
    function: ast.FunctionDef,
    pointer_roles: set[str],
    local_bindings: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Independently check source-order bindings for replayed memory effects."""

    arguments = tuple(argument.arg for argument in function.args.args)
    formals = set(arguments)
    constexpr = {
        argument.arg
        for argument in function.args.args
        if isinstance(argument.annotation, ast.Attribute)
        and isinstance(argument.annotation.value, ast.Name)
        and argument.annotation.value.id == "tl"
        and argument.annotation.attr == "constexpr"
    }
    allowed_tl = set(_TRITON_SOURCE_TL_OPERATIONS)
    allowed_bare = set(_TRITON_SOURCE_BARE_CALLS)
    reads: set[str] = set()
    writes: set[str] = set()
    visited_calls: set[int] = set()
    guard_dependencies: dict[str, frozenset[str]] = {}
    # Snapshot each binding at its definition so a later statement cannot close it.
    opaque, open_value = object(), object()

    def operation(node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "tl":
            return node.attr
        return None

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
                name = operation(node.func)
                admitted = name in allowed_tl or (
                    isinstance(node.func, ast.Name)
                    and node.func.id in allowed_bare
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
        if isinstance(node, ast.BinOp):
            return may_resolve_to_none(node.left, values) or may_resolve_to_none(node.right, values)
        if isinstance(node, ast.UnaryOp):
            return may_resolve_to_none(node.operand, values)
        if isinstance(node, (ast.Tuple, ast.List)):
            return any(may_resolve_to_none(value, values) for value in node.elts)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "min":
            return any(may_resolve_to_none(value, values) for value in node.args) or any(
                keyword.arg == "default" and may_resolve_to_none(keyword.value, values)
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
            if node.id in pointer_roles:
                return frozenset((node.id,))
            return aliases.get(node.id, frozenset())
        if isinstance(node, ast.BinOp):
            return roots(node.left, aliases) | roots(node.right, aliases)
        if isinstance(node, ast.UnaryOp):
            return roots(node.operand, aliases)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "to":
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
            if operation(node.func) in {"load", "store"}:
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
        if isinstance(node, ast.Call) and operation(node.func) in {"load", "store"}:
            return False
        return closed(node, values)

    def condition(node: ast.AST, known: set[str]) -> tuple[bool, bool, bool]:
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
            left, right = condition(node.left, known), condition(node.right, known)
            return (
                left[0] and right[0] and not (left[2] or right[2]),
                left[1] or right[1],
                left[2] or right[2],
            )
        if isinstance(node, ast.BoolOp):
            parts = [condition(value, known) for value in node.values]
            return (
                all(part[0] and not part[2] for part in parts),
                any(part[1] for part in parts),
                any(part[2] for part in parts),
            )
        if isinstance(node, ast.Compare):
            parts = [condition(node.left, known), *(condition(value, known) for value in node.comparators)]
            modules = [index for index, part in enumerate(parts) if part[2]]
            if any(not part[0] for part in parts) or len(modules) > 1:
                return False, False, False
            if not modules:
                return True, any(part[1] for part in parts), False
            if len(parts) != 2:
                return False, False, False
            dependent = any(part[1] for index, part in enumerate(parts) if index != modules[0])
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
                isinstance(item, ast.Call) and operation(item.func) in {"load", "store"}
                for item in ast.walk(node)
            )
            and closed(node, values)
        )

    def scoped_guards(
        guarded: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        return {key: dict(bindings) for key, bindings in guarded.items()}

    def merge_guarded(
        guarded: dict[str, dict[str, object]],
        branches: tuple[dict[str, dict[str, object]], ...],
    ) -> None:
        for key in tuple(guarded):
            if any(key not in branch for branch in branches):
                guarded.pop(key)
                continue
            bindings = guarded[key]
            for item in tuple(bindings):
                states = tuple(branch.get(key, {}).get(item, open_value) for branch in branches)
                if any(state is open_value for state in states):
                    bindings.pop(item)
                elif any(state is None for state in states):
                    bindings[item] = None
                else:
                    bindings[item] = opaque

    def update_guarded(
        guarded: dict[str, dict[str, object]], bound: tuple[str, ...], state: object
    ) -> None:
        bound_names = frozenset(bound)
        for key in tuple(guarded):
            if guard_dependencies.get(key, frozenset()) & bound_names:
                guarded.pop(key)
                continue
            bindings = guarded[key]
            for item in bound:
                if item in bindings:
                    if state is open_value:
                        bindings.pop(item)
                    else:
                        bindings[item] = state

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
        # Preserve only entry bindings whose active state is closed on every path.
        for item, previous in initial_values.items():
            if any(branch_values[item] != previous for branch_values, _, _ in branches):
                states = tuple(branch_values[item] for branch_values, _, _ in branches)
                if any(state is open_value for state in states):
                    values[item] = open_value
                elif any(state is None for state in states):
                    values[item] = None
                else:
                    values[item] = opaque
        for item, previous in initial_aliases.items():
            if any(branch_aliases.get(item) != previous for _, branch_aliases, _ in branches):
                aliases.pop(item, None)
        known.intersection_update(*(branch_known for _, _, branch_known in branches))

    def names(target: ast.AST) -> tuple[str, ...] | None:
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, (ast.Tuple, ast.List)):
            result: list[str] = []
            for item in target.elts:
                item_names = names(item)
                if item_names is None:
                    return None
                result.extend(item_names)
            return tuple(result) if result else None
        return None

    def assigned_values(
        target: ast.AST, value: ast.AST
    ) -> tuple[tuple[str, ast.AST], ...] | None:
        if isinstance(target, ast.Name):
            return ((target.id, value),)
        if (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
        ):
            result: list[tuple[str, ast.AST]] = []
            for item, source in zip(target.elts, value.elts):
                assigned = assigned_values(item, source)
                if assigned is None:
                    return None
                result.extend(assigned)
            return tuple(result)
        return None

    def calls(node: ast.Call, values: dict[str, object], aliases: dict[str, frozenset[str]]) -> None:
        if id(node) in visited_calls:
            return
        visited_calls.add(id(node))
        name = operation(node.func)
        if name is None:
            if isinstance(node.func, ast.Name) and node.func.id in allowed_bare:
                if node.func.id in local_bindings:
                    raise ReplayError(f"SOURCE_PROFILE: supported callable {node.func.id} is shadowed locally")
                return
            if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
                if not closed(node.func.value, values):
                    raise ReplayError(
                        f"SOURCE_PROFILE: .to receiver is not a supported Triton expression at line {node.lineno}"
                    )
                return
            label = node.func.id if isinstance(node.func, ast.Name) else ast.dump(node.func)
            raise ReplayError(f"call {label} is outside the supported source subset at line {node.lineno}")
        if name not in allowed_tl:
            raise ReplayError(
                f"Triton operation tl.{name} is outside the supported source subset at line {node.lineno}"
            )
        if name not in {"load", "store"}:
            return
        positional_count = 1 if name == "load" else 2
        allowed_keywords = {"mask", "other"} if name == "load" else {"mask"}
        if len(node.args) != positional_count:
            raise ReplayError(
                f"SOURCE_PROFILE: tl.{name} requires exactly {positional_count} positional operands at line {node.lineno}"
            )
        keyword_names = [keyword.arg for keyword in node.keywords]
        if (
            any(item is None for item in keyword_names)
            or len(set(keyword_names)) != len(keyword_names)
            or not set(keyword_names).issubset(allowed_keywords)
        ):
            raise ReplayError(
                f"SOURCE_PROFILE: tl.{name} has unsupported or duplicate keywords at line {node.lineno}"
            )
        keyword_values = {keyword.arg: keyword.value for keyword in node.keywords}
        if name == "load" and "other" in keyword_values:
            mask = keyword_values.get("mask")
            if mask is None or may_resolve_to_none(mask, values):
                raise ReplayError(
                    f"SOURCE_PROFILE: tl.load other requires a non-None mask at line {node.lineno}"
                )
        operands = (*node.args, *(keyword.value for keyword in node.keywords))
        nested_load = (
            name == "store"
            and isinstance(node.args[1], ast.Call)
            and operation(node.args[1].func) == "load"
        )
        if nested_load:
            calls(node.args[1], values, aliases)
        if any(
            not memory_value(value, values)
            for index, value in enumerate(operands)
            if not (nested_load and index == 1)
        ):
            raise ReplayError(f"SOURCE_PROFILE: tl.{name} has an open operand at line {node.lineno}")
        pointer_roots = roots(node.args[0], aliases)
        if len(pointer_roots) != 1:
            raise ReplayError(
                f"tl.{name} pointer has ambiguous formal roots at line {node.lineno}"
            )
        if name == "load":
            reads.add(next(iter(pointer_roots)))
        else:
            writes.add(next(iter(pointer_roots)))

    def block(
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
                    raise ReplayError(
                        f"SOURCE_PROFILE: if predicate is not compile-time closed at line {node.lineno}"
                    )
                key = ast.dump(node.test, include_attributes=False)
                guard_dependencies[key] = frozenset(
                    item.id
                    for item in ast.walk(node.test)
                    if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                )
                initial_values, initial_aliases = dict(values), dict(aliases)
                guarded.setdefault(key, {})
                body_guarded = scoped_guards(guarded)
                body_guarded.setdefault(key, {})
                body_values = dict(values)
                body_values.update(guarded.get(key, {}))
                body_values, body_aliases, body_known = block(
                    node.body, body_values, dict(aliases), set(known), body_guarded
                )
                else_guarded = scoped_guards(guarded)
                else_guarded.setdefault(key, {})
                else_values, else_aliases, else_known = block(
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
                for item in body_added - set(body_aliases):
                    if (
                        key in guarded
                        and key in body_guarded
                        and key in else_guarded
                        and body_values[item] is not open_value
                    ):
                        guarded.setdefault(key, {})[item] = body_values[item]
                for item in (body_added & else_added) - set(body_aliases) - set(else_aliases):
                    if body_values[item] is open_value or else_values[item] is open_value:
                        values[item] = open_value
                    elif body_values[item] is None or else_values[item] is None:
                        values[item] = None
                    else:
                        values[item] = opaque
                    guarded.get(key, {}).pop(item, None)
                    if item in body_known and item in else_known:
                        known.add(item)
                continue
            if isinstance(node, ast.For):
                static = isinstance(node.iter, ast.Call) and operation(node.iter.func) == "static_range"
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
                        or static
                    )
                    or not 1 <= len(node.iter.args) <= 3
                    or any(
                        not static_scalar(value, known) if static
                        else not range_scalar(value, values, aliases)
                        for value in node.iter.args
                    )
                ):
                    raise ReplayError(
                        f"SOURCE_PROFILE: for-loop control flow is outside the supported source subset at line {node.lineno}"
                    )
                initial_values = dict(values)
                initial_aliases = dict(aliases)
                initial_known = set(known)
                initial_guarded = scoped_guards(guarded)
                loop_guarded = scoped_guards(guarded)
                loop_values = dict(values)
                loop_values[node.target.id] = opaque
                loop_aliases = dict(aliases)
                loop_aliases.pop(node.target.id, None)
                loop_known = set(known)
                loop_known.discard(node.target.id)
                update_guarded(loop_guarded, (node.target.id,), opaque)
                loop_values, loop_aliases, loop_known = block(
                    node.body,
                    loop_values,
                    loop_aliases,
                    loop_known,
                    loop_guarded,
                )
                if (
                    any(
                        loop_values.get(item, open_value) is not state
                        for item, state in initial_values.items()
                    )
                    or any(
                        (item in loop_aliases) != (item in initial_aliases)
                        or loop_aliases.get(item) != initial_aliases.get(item)
                        for item in initial_values
                    )
                    or any(
                        (item in loop_known) != (item in initial_known)
                        for item in initial_values
                    )
                    or any(
                        key not in loop_guarded
                        or any(
                            loop_guarded[key].get(item, open_value) is not state
                            for item, state in bindings.items()
                        )
                        for key, bindings in initial_guarded.items()
                    )
                ):
                    raise ReplayError(
                        f"SOURCE_PROFILE: for-loop control flow is outside the supported source subset at line {node.lineno}"
                    )
                continue
            assignment = None
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    raise ReplayError(
                        f"SOURCE_PROFILE: assignment is outside the supported source subset at line {node.lineno}"
                    )
                bound = names(node.targets[0])
                if bound is None:
                    raise ReplayError(
                        f"SOURCE_PROFILE: assignment is outside the supported source subset at line {node.lineno}"
                    )
                assignment = (bound, node.value)
            elif isinstance(node, ast.AnnAssign):
                if node.value is None:
                    raise ReplayError(
                        f"SOURCE_PROFILE: assignment is outside the supported source subset at line {node.lineno}"
                    )
                bound = names(node.target)
                if bound is None:
                    raise ReplayError(
                        f"SOURCE_PROFILE: assignment is outside the supported source subset at line {node.lineno}"
                    )
                assignment = (bound, node.value)
            elif isinstance(node, ast.AugAssign) and not isinstance(node.target, ast.Name):
                raise ReplayError(
                    f"SOURCE_PROFILE: assignment is outside the supported source subset at line {node.lineno}"
                )
            if any(isinstance(nested, (ast.IfExp, ast.NamedExpr)) for nested in ast.walk(node)):
                raise ReplayError(
                    f"SOURCE_PROFILE: assignment is outside the supported source subset at line {node.lineno}"
                )
            for nested in ast.walk(node):
                if isinstance(nested, ast.Call):
                    calls(nested, values, aliases)
            if assignment is not None:
                bound, value = assignment
                bindings = assigned_values(
                    node.targets[0] if isinstance(node, ast.Assign) else node.target,
                    value,
                )
                if bindings is None:
                    bindings = tuple((item, value) for item in bound)
                contains_memory = any(
                    isinstance(child, ast.Call) and operation(child.func) in {"load", "store"}
                    for child in ast.walk(value)
                )
                value_closed = closed(value, values)
                if contains_memory and not value_closed:
                    raise ReplayError(f"SOURCE_PROFILE: assignment has an open value at line {node.lineno}")
                for item, bound_value in bindings:
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
                    values[item] = state
                    if pointer_roots and bound_closed:
                        aliases[item] = pointer_roots
                    else:
                        aliases.pop(item, None)
                    update_guarded(guarded, (item,), state)
                known.difference_update(bound)
                constant_value = condition(value, known)
                if constant_value[0] and not constant_value[2]:
                    known.update(bound)
                continue
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                if (
                    node.target.id not in values and node.target.id not in formals
                ) or not closed(node.value, values):
                    raise ReplayError(f"SOURCE_PROFILE: assignment has an open value at line {node.lineno}")
                pointer_tainted = (
                    values.get(node.target.id, opaque) is open_value
                    or bool(aliases.get(node.target.id))
                    or bool(range_roots(node.value, aliases))
                )
                values[node.target.id] = (
                    open_value
                    if pointer_tainted
                    else None
                    if may_resolve_to_none(node.target, values)
                    or may_resolve_to_none(node.value, values)
                    else opaque
                )
                aliases.pop(node.target.id, None)
                update_guarded(guarded, (node.target.id,), values[node.target.id])
                known.discard(node.target.id)
        return values, aliases, known

    block(function.body, dict.fromkeys(formals, opaque), {}, constexpr, {})
    if not reads and not writes:
        raise ReplayError("kernel source contains no supported Triton memory effects")
    return arguments, tuple(sorted(reads)), tuple(sorted(writes))
