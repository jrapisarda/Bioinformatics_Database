"""
Rule definitions and configuration utilities.

This module defines a default set of statistical rules used to score
gene pairs and provides helper functions for loading, modifying and
saving rule configurations. Each rule is represented as a dictionary
with the following keys:

* ``name`` – a unique identifier for the rule.
* ``condition`` – a Python expression evaluated against a pair's
  attributes. The expression should return ``True`` when the rule is
  satisfied. Supported operators include arithmetic operators, logical
  operators (``and``, ``or``, ``not``) and comparison operators. The
  built‐in functions ``abs``, ``min``, ``max``, ``sqrt`` and
  ``math`` are also available.
* ``weight`` – a floating point number representing the relative
  importance of the rule when aggregating rule scores.
* ``explanation`` – a human‐readable description of what the rule is
  intended to capture.

Rules can be stored on disk in JSON format using
:func:`save_rules_config` and later loaded with
:func:`load_rules_config`. When adding new rules via the web interface
or programmatically, be sure to assign a unique ``name``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

DEFAULT_RULES: List[Dict[str, object]] = [
    {
        "name": "statistical_significance",
        "condition": "(p_ss < 0.05) and (p_soth < 0.01)",
        "weight": 0.25,
        "explanation": "Both conditions show statistical significance",
    },
    {
        "name": "effect_size",
        "condition": "(abs_dz_ss > 0.3) and (abs_dz_soth > 1.0)",
        "weight": 0.30,
        "explanation": "Large effect sizes in both conditions",
    },
    {
        "name": "z_score_strength",
        "condition": "(abs(dz_ss_z) > 1.5) and (abs(dz_soth_z) > 3.0)",
        "weight": 0.20,
        "explanation": "Strong z-scores across conditions",
    },
    {
        "name": "fdr_correction",
        "condition": "(q_ss is not None and q_ss < 0.2) and (q_soth is not None and q_soth < 0.01)",
        "weight": 0.15,
        "explanation": "Survives FDR correction",
    },
    {
        "name": "consistency",
        "condition": "((dz_ss_I2 is not None and dz_ss_I2 < 50) or (dz_soth_I2 is not None and dz_soth_I2 < 75))",
        "weight": 0.10,
        "explanation": "Low heterogeneity across studies",
    },
]


def load_rules_config(config_path: Path) -> List[Dict[str, object]]:
    """Load a list of rule definitions from a JSON file.

    If the file does not exist or cannot be parsed, the default rules
    are returned.

    Parameters
    ----------
    config_path : Path
        Path to the JSON file storing rule definitions.

    Returns
    -------
    List[dict]
        A list of rule dictionaries.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        return DEFAULT_RULES.copy()
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            rules = json.load(fh)
        # Basic validation: ensure each rule has required keys
        validated = []
        for rule in rules:
            if not all(k in rule for k in ("name", "condition", "weight", "explanation")):
                continue
            validated.append(rule)
        return validated if validated else DEFAULT_RULES.copy()
    except Exception:
        return DEFAULT_RULES.copy()


def save_rules_config(config_path: Path, rules: List[Dict[str, object]]) -> None:
    """Save a list of rule definitions to a JSON file.

    Parameters
    ----------
    config_path : Path
        Destination path for the configuration file. Parent
        directories are created if necessary.
    rules : list of dict
        List of rule definitions to serialise.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(rules, fh, indent=2)


def add_rule(
    rules: List[Dict[str, object]], name: str, condition: str, weight: float, explanation: str
) -> List[Dict[str, object]]:
    """Add a new rule to the existing list of rules.

    Raises a ``ValueError`` if a rule with the same name already exists.
    
    Parameters
    ----------
    rules : list of dict
        Current rule set.
    name : str
        Unique identifier for the new rule.
    condition : str
        Python expression to evaluate against pair attributes.
    weight : float
        Importance weight of the rule.
    explanation : str
        Human readable description of the rule.

    Returns
    -------
    list of dict
        Updated list of rules.
    """
    if any(rule["name"] == name for rule in rules):
        raise ValueError(f"A rule named '{name}' already exists.")
    new_rule = {
        "name": name,
        "condition": condition,
        "weight": float(weight),
        "explanation": explanation,
    }
    return rules + [new_rule]