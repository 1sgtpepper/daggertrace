#!/usr/bin/env python3
"""Check the package's small set of import-direction boundaries."""

from __future__ import annotations

import ast
from pathlib import Path


_PACKAGE = "daggertrace"
_ROOT = Path(__file__).resolve().parents[1] / "src" / _PACKAGE
_REPLAY = f"{_PACKAGE}.replay"
_REPLAY_FORBIDDEN = frozenset(
    {
        "torch",
        "triton",
        f"{_PACKAGE}.capture",
        f"{_PACKAGE}.liger_capture",
        f"{_PACKAGE}.liger_target",
        f"{_PACKAGE}.art_capture",
        f"{_PACKAGE}.targets",
        f"{_PACKAGE}.target_support",
        f"{_PACKAGE}.liger_packet",
        f"{_PACKAGE}.art_target",
        f"{_PACKAGE}.memory_model_target",
        f"{_PACKAGE}.model",
        f"{_PACKAGE}.model_core",
        f"{_PACKAGE}.ownership",
        f"{_PACKAGE}.ownership_common",
        f"{_PACKAGE}.ownership_v2",
        f"{_PACKAGE}.ownership_v2_runtime",
        f"{_PACKAGE}.ownership_v2_validate",
    }
)
_CORE = frozenset(
    {
        f"{_PACKAGE}.canonical",
        f"{_PACKAGE}.errors",
        f"{_PACKAGE}.model",
        f"{_PACKAGE}.model_core",
        f"{_PACKAGE}.ownership",
        f"{_PACKAGE}.ownership_common",
        f"{_PACKAGE}.ownership_v2",
        f"{_PACKAGE}.ownership_v2_runtime",
        f"{_PACKAGE}.ownership_v2_validate",
    }
)
_CLI = frozenset(
    {
        f"{_PACKAGE}.cli",
        f"{_PACKAGE}.cli_audit",
        f"{_PACKAGE}.cli_output",
        f"{_PACKAGE}.cli_replay",
    }
)
_TARGET_ROUTER = f"{_PACKAGE}.targets"
_TARGET_ADAPTERS = frozenset(
    {
        f"{_PACKAGE}.art_capture",
        f"{_PACKAGE}.art_target",
        f"{_PACKAGE}.liger_capture",
        f"{_PACKAGE}.liger_packet",
        f"{_PACKAGE}.liger_target",
        f"{_PACKAGE}.memory_model_target",
    }
)
_TARGET_ADAPTER_FAMILIES = {
    f"{_PACKAGE}.art_capture": "ART",
    f"{_PACKAGE}.art_target": "ART",
    f"{_PACKAGE}.liger_capture": "Liger",
    f"{_PACKAGE}.liger_packet": "Liger",
    f"{_PACKAGE}.liger_target": "Liger",
    f"{_PACKAGE}.memory_model_target": "Triton LayerNorm",
}


def _module_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((_PACKAGE, *parts)) if parts else _PACKAGE


def _resolve_from(
    module: str,
    is_package: bool,
    node: ast.ImportFrom,
    modules: set[str],
) -> set[str]:
    if not node.level:
        if not node.module:
            return set()
        candidates = {node.module}
        candidates.update(
            candidate
            for alias in node.names
            if (candidate := f"{node.module}.{alias.name}") in modules
        )
        return candidates
    parts = module.split(".")
    ancestor_parts = len(parts) - node.level + (1 if is_package else 0)
    anchor = parts[:ancestor_parts]
    if node.module:
        anchor.extend(node.module.split("."))
    base = ".".join(anchor)
    candidates = {base} if node.module else set()
    if node.module is None:
        candidates.update(f"{base}.{alias.name}" for alias in node.names)
    return candidates & modules


class _ModuleImports(ast.NodeVisitor):
    """Collect imports executed while a module is initialized, not lazy imports."""

    def __init__(self, module: str, is_package: bool, modules: set[str]) -> None:
        self._module = module
        self._is_package = is_package
        self._modules = modules
        self.imports: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.update(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.update(_resolve_from(self._module, self._is_package, node, self._modules))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _graph(root: Path) -> dict[str, set[str]]:
    paths = sorted(root.rglob("*.py"))
    modules = {_module_name(root, path) for path in paths}
    graph: dict[str, set[str]] = {}
    for path in paths:
        module = _module_name(root, path)
        visitor = _ModuleImports(module, path.name == "__init__.py", modules)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        graph[module] = visitor.imports
    return graph


def _reachable(graph: dict[str, set[str]], start: str) -> dict[str, tuple[str, ...]]:
    paths = {start: (start,)}
    pending = [start]
    while pending:
        current = pending.pop()
        for dependency in sorted(graph.get(current, ())):
            if dependency not in paths:
                paths[dependency] = (*paths[current], dependency)
                pending.append(dependency)
    return paths


def _cycles(graph: dict[str, set[str]]) -> tuple[tuple[str, ...], ...]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(module: str) -> None:
        nonlocal index
        indices[module] = index
        lowlinks[module] = index
        index += 1
        stack.append(module)
        active.add(module)
        for dependency in sorted(graph[module]):
            if dependency not in graph:
                continue
            if dependency not in indices:
                visit(dependency)
                lowlinks[module] = min(lowlinks[module], lowlinks[dependency])
            elif dependency in active:
                lowlinks[module] = min(lowlinks[module], indices[dependency])
        if lowlinks[module] != indices[module]:
            return
        component = []
        while True:
            dependency = stack.pop()
            active.remove(dependency)
            component.append(dependency)
            if dependency == module:
                break
        if len(component) > 1 or module in graph[module]:
            components.append(tuple(sorted(component)))

    for module in sorted(graph):
        if module not in indices:
            visit(module)
    return tuple(components)


def _forbidden_path(
    graph: dict[str, set[str]], start: str, forbidden: frozenset[str]
) -> tuple[str, ...] | None:
    for dependency, path in _reachable(graph, start).items():
        if dependency != start and dependency in forbidden:
            return path
    return None


def _replay_forbidden(dependency: str) -> bool:
    return dependency in _REPLAY_FORBIDDEN or dependency.startswith(("torch.", "triton."))


def check(root: Path = _ROOT) -> tuple[str, ...]:
    """Return boundary violations for a package root without changing it."""

    graph = _graph(root)
    problems = [
        "top-level internal import cycle: " + ", ".join(component)
        for component in _cycles(graph)
    ]

    if _REPLAY in graph:
        for dependency, path in _reachable(graph, _REPLAY).items():
            if dependency != _REPLAY and _replay_forbidden(dependency):
                problems.append("dependency-light replay reaches " + " -> ".join(path))
                break

    upward = _CLI | _TARGET_ADAPTERS | {_TARGET_ROUTER}
    for module in sorted(_CORE & graph.keys()):
        path = _forbidden_path(graph, module, upward)
        if path is not None:
            problems.append("core layer reaches upward dependency " + " -> ".join(path))

    support = f"{_PACKAGE}.target_support"
    if support in graph:
        path = _forbidden_path(graph, support, _TARGET_ADAPTERS | {_TARGET_ROUTER})
        if path is not None:
            problems.append("target support reaches routing or an adapter " + " -> ".join(path))

    if _TARGET_ROUTER in graph:
        path = _forbidden_path(graph, _TARGET_ROUTER, _TARGET_ADAPTERS)
        if path is not None:
            problems.append("target router eagerly imports adapter " + " -> ".join(path))

    for module in sorted(_TARGET_ADAPTERS & graph.keys()):
        family = _TARGET_ADAPTER_FAMILIES[module]
        forbidden = frozenset({_TARGET_ROUTER}) | frozenset(
            dependency
            for dependency in _TARGET_ADAPTERS
            if _TARGET_ADAPTER_FAMILIES[dependency] != family
        )
        path = _forbidden_path(graph, module, forbidden)
        if path is not None:
            problems.append("target adapter crosses family boundary " + " -> ".join(path))

    return tuple(problems)


def main() -> int:
    problems = check()
    if not problems:
        return 0
    for problem in problems:
        print(f"architecture: {problem}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
