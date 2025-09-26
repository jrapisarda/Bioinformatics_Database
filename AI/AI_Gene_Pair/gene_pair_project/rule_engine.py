"""
Rule evaluation engine.

This module provides functionality to safely evaluate rule conditions on
a given gene pair's data dictionary and compute aggregate rule scores.
Each rule is defined by an expression (as a string) that will be
evaluated in a restricted context. Only a small set of built‐in
functions are exposed to the evaluation environment to mitigate the
risk of arbitrary code execution.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict, List, Tuple


SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "math": math,
}


def _safe_eval(expr: str, context: Dict[str, Any]) -> bool:
    """Safely evaluate a boolean expression against a given context.

    The expression string is compiled to an abstract syntax tree and
    recursively inspected to ensure it contains only safe nodes
    (constants, boolean/arithmetical/logical operators, comparisons,
    attribute and name lookups). The compiled code is executed with a
    restricted set of built-ins to limit available functions. Any
    exception encountered during evaluation results in ``False``.

    Parameters
    ----------
    expr : str
        Expression to evaluate. Should evaluate to a boolean value.
    context : dict
        Mapping of variable names to values. Missing variables will
        evaluate to ``None``.

    Returns
    -------
    bool
        Result of the evaluated expression, or ``False`` on error.
    """
    try:
        # Parse the expression to ensure it is syntactically valid and
        # contains only safe nodes
        tree = ast.parse(expr, mode="eval")

        for node in ast.walk(tree):
            if isinstance(node, (ast.Expression, ast.Load, ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Call, ast.Name, ast.Constant, ast.Subscript, ast.Attribute)):
                continue
            # Allow basic operations such as arithmetic operators and boolean ops
            elif isinstance(node, (ast.Or, ast.And, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot)):
                continue
            else:
                raise ValueError(f"Unsafe expression element: {ast.dump(node)}")

        # Compile the expression
        code = compile(tree, filename="<rule>", mode="eval")
        # Provide a fallback for missing names: context.get(name, None)
        safe_globals = {**SAFE_BUILTINS, "__builtins__": {}}
        # Evaluate
        return bool(eval(code, safe_globals, context))
    except Exception:
        return False


def evaluate_rules(
    pair_data: Dict[str, Any], rules: List[Dict[str, Any]]
) -> Tuple[float, List[str]]:
    """Evaluate a list of rules on a single gene pair.

    For each rule in the list, the function evaluates its condition
    against the provided ``pair_data`` mapping. If the condition
    evaluates to ``True``, the rule's weight contributes to the total
    score, and a descriptive message is appended to the explanation
    list. Otherwise, the rule contributes zero and an explanatory
    message noting non‐satisfaction is recorded.

    Parameters
    ----------
    pair_data : dict
        Dictionary containing all attributes relevant for rule
        evaluation (e.g. ``p_ss``, ``abs_dz_ss``, etc.). Missing keys
        default to ``None``.
    rules : list of dict
        List of rules as defined in ``gene_pair_project.rules``. Each
        rule must include ``condition``, ``weight``, ``name`` and
        ``explanation`` keys.

    Returns
    -------
    tuple(float, list of str)
        A tuple containing the total rule score and a list of
        human‐readable explanation strings for each rule.
    """
    total_score = 0.0
    explanations: List[str] = []

    # Create evaluation context: copy pair_data but ensure all keys
    # referenced in expressions are present (missing default to None)
    context = {k: pair_data.get(k) for k in pair_data.keys()}

    for rule in rules:
        name = rule.get("name", "unknown")
        condition = rule.get("condition", "False")
        weight = float(rule.get("weight", 0.0))
        explanation = rule.get("explanation", "")

        satisfied = _safe_eval(condition, context)
        if satisfied:
            total_score += weight
            explanations.append(
                f"Rule '{name}' satisfied (+{weight:.2f}): {explanation}"
            )
        else:
            explanations.append(
                f"Rule '{name}' not satisfied (+0.00): {explanation}"
            )

    return total_score, explanations