"""`lab lint` — will the entrypoint accept the override keys a submit or sweep is about to pass?

A sweep appends each point as ``key=value`` argv tokens (and a sharded sweep adds ``seeds=…``).
An entrypoint that adopts the handshake calls ``get_overrides(known={...})``: that set is the
whole truth — every other key makes it exit non-zero before doing any work. So when the set can
be read statically from the source, lint checks against it. The old check — is the key's literal
string anywhere in the file? — stays as the fallback for legacy entrypoints that declare
nothing; it is the one that let ``--grid gain=2`` pass on ``example_search.py`` (where ``gain``
is a ``lab.search`` grid axis, not an override) and the job fail with "unknown config override".

Static and conservative: anything this cannot read for certain (a ``known`` computed at runtime,
``known=None``, a syntax error) returns ``None`` and the grep decides.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import Any

from lab.experiment import unreferenced_keys

_COLLECTIONS = frozenset({"set", "frozenset", "list", "tuple", "sorted"})


def _call_names(tree: ast.AST) -> set[str]:
    """The bare names under which ``get_overrides`` may be called (``import … as go``)."""
    funcs = {"get_overrides"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "lab.experiment":
            funcs |= {a.asname or a.name for a in node.names if a.name == "get_overrides"}
    return funcs


def _is_get_overrides(func: ast.expr, funcs: set[str]) -> bool:
    if isinstance(func, ast.Name):
        return func.id in funcs
    # `lab.experiment.get_overrides(...)`, `e.get_overrides(...)` after `import … as e`
    return isinstance(func, ast.Attribute) and func.attr == "get_overrides"


def _strings(expr: ast.expr, names: dict[str, ast.expr], depth: int = 0) -> set[str] | None:
    """The string set ``expr`` evaluates to, if that is statically certain; else ``None``."""
    if depth > 8:
        return None
    if isinstance(expr, (ast.Set, ast.List, ast.Tuple)):
        out = set()
        for elt in expr.elts:
            if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                return None
            out.add(elt.value)
        return out
    if isinstance(expr, ast.Dict):  # iterating a dict yields its keys
        return _strings(ast.Set(elts=[k for k in expr.keys if k is not None]), names, depth + 1)
    if isinstance(expr, ast.Name) and expr.id in names:
        return _strings(names[expr.id], names, depth + 1)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
        left, right = _strings(expr.left, names, depth + 1), _strings(expr.right, names, depth + 1)
        return None if left is None or right is None else left | right
    if isinstance(expr, ast.Call) and not expr.keywords:
        func = expr.func
        if isinstance(func, ast.Name) and func.id in _COLLECTIONS and len(expr.args) == 1:
            return _strings(expr.args[0], names, depth + 1)
        if isinstance(func, ast.Attribute) and func.attr == "keys" and not expr.args:
            return _strings(func.value, names, depth + 1)
    return None


def declared_keys(source: str) -> set[str] | None:
    """The override keys the entrypoint's ``get_overrides(known=...)`` call(s) accept — their
    union — or ``None`` when there is no such call or any ``known`` is not statically readable."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    names: dict[str, ast.expr] = {}
    for node in ast.walk(tree):  # simple `NAME = <literal>` bindings a `known=NAME` may use
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                names[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            names[node.target.id] = node.value
    funcs = _call_names(tree)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _is_get_overrides(n.func, funcs)]
    if not calls:
        return None
    declared: set[str] = set()
    for call in calls:
        known: ast.expr | None = next((k.value for k in call.keywords if k.arg == "known"), None)
        if known is None and call.args:
            known = call.args[0]
        found = _strings(known, names) if known is not None else None
        if found is None:
            return None  # no schema here (or a computed one): nothing refuses keys statically
        declared |= found
    return declared


def lint_keys(source: str, keys: Iterable[str]) -> dict[str, Any]:
    """``{check, declared_keys, missing_keys}`` for the override ``keys`` about to be passed.

    ``check`` is ``"declared"`` when ``get_overrides(known=...)`` could be read (``missing_keys``
    are keys it will refuse at runtime) and ``"heuristic"`` otherwise (keys whose literal string
    never appears in the source — a grep, by design). The heuristic leaves a grid ``seed`` alone:
    nothing refuses it in a legacy script, and the lab consumes it as ``$LAB_SEED``.
    """
    wanted = sorted(set(keys))
    declared = declared_keys(source)
    if declared is not None:
        return {
            "check": "declared",
            "declared_keys": sorted(declared),
            "missing_keys": [k for k in wanted if k not in declared],
        }
    return {
        "check": "heuristic",
        "declared_keys": None,
        "missing_keys": unreferenced_keys(source, [k for k in wanted if k != "seed"]),
    }
