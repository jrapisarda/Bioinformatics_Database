"""
Data loading and feature engineering utilities for gene pair analysis.

This module defines helper functions for reading gene pair meta‐
analysis results from a variety of formats and deriving derived
statistics for downstream machine learning models and rule evaluation.

The input data should contain the required fields defined in
`gene-pair-ml-requirements.md`, namely:

* ``pair_id`` (string) – Unique identifier for each gene pair.
* ``GeneAName`` (string) – First gene name.
* ``GeneBName`` (string) – Second gene name.
* ``dz_ss_mean`` (float) – Effect size mean for sepsis condition.
* ``dz_soth_mean`` (float) – Effect size mean for septic shock condition.
* ``p_ss`` (float) – P‐value for sepsis condition.
* ``p_soth`` (float) – P‐value for septic shock condition.
* ``abs_dz_ss`` (float) – Absolute effect size for sepsis.
* ``abs_dz_soth`` (float) – Absolute effect size for septic shock.

Optional enhancement fields (standard errors, confidence interval
bounds, heterogeneity statistics, q‐values, etc.) can also be
supplied. When present, they will be incorporated into engineered
features; otherwise missing values will be gracefully handled.

Examples
--------

>>> from gene_pair_project.data_processing import load_data, engineer_features
>>> df = load_data("pairs.xlsx")
>>> df_features = engineer_features(df)

"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Dict, Tuple, Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

# List of mandatory input fields based on requirements document
MANDATORY_FIELDS = [
    "pair_id",
    "GeneAName",
    "GeneBName",
    "dz_ss_mean",
    "dz_soth_mean",
    "p_ss",
    "p_soth",
    "abs_dz_ss",
    "abs_dz_soth",
]


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load gene pair data from CSV, Excel or JSON files.

    Parameters
    ----------
    file_path : str or Path
        Location of the input file. CSV (``.csv``), Excel (``.xlsx``,
        ``.xls``) and JSON (``.json``) formats are supported. For JSON
        files, the data must be a list of objects where each object
        corresponds to a pair.

    Returns
    -------
    DataFrame
        Parsed dataset containing all rows from the input file. Raises
        ``ValueError`` if required columns are missing.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file {file_path} not found")

    # Detect format and load accordingly
    if file_path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    elif file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        df = pd.json_normalize(data)
    else:
        raise ValueError(
            f"Unsupported file type '{file_path.suffix}'. Please provide CSV, Excel or JSON."
        )

    # Check required columns
    missing = [col for col in MANDATORY_FIELDS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input data is missing required fields: {', '.join(missing)}."
        )

    return df


def _compute_z_score(p_value: float, effect: float) -> float:
    """Approximate a z-score from a two-tailed p-value and effect sign.

    This function infers the z-score corresponding to a two-tailed
    p-value assuming a normal distribution and applies the sign of the
    provided effect size. If the p-value is zero or NaN the z-score is
    set to zero.
    """
    try:
        if p_value is None or math.isnan(p_value) or p_value <= 0:
            return 0.0
        z = norm.ppf(1 - p_value / 2)
        return z if effect >= 0 else -z
    except Exception:
        return 0.0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate additional features from raw meta-analysis statistics.

    The resulting DataFrame contains all original columns plus new
    columns used by the machine learning models and rule engine:

    * ``combined_p_value`` – geometric mean of p-values across
      conditions
    * ``combined_effect_size`` – geometric mean of absolute effect sizes
      across conditions
    * ``dz_ss_z`` and ``dz_soth_z`` – approximated z-scores for
      sepsis and septic shock effect sizes
    * ``combined_z_score`` – Euclidean norm of the individual z-scores
    * ``statistical_power`` – simple power indicator
    * ``delta_effect_size`` – difference between septic shock and sepsis
      effect sizes
    * ``abs_delta_effect_size`` – absolute difference between effect
      sizes
    * ``baseline_similarity`` – placeholder for similarity to the
      positive control (to be updated later)

    Parameters
    ----------
    df : DataFrame
        Input data as loaded by :func:`load_data`.

    Returns
    -------
    DataFrame
        New DataFrame containing the engineered features alongside
        original columns. Missing numeric values are filled with
        ``NaN``; downstream components should handle these gracefully.
    """
    df = df.copy()

    # Combined p-value: geometric mean of p_ss and p_soth
    df["combined_p_value"] = np.sqrt(df["p_ss"].astype(float) * df["p_soth"].astype(float))

    # Combined effect size: geometric mean of absolute effect sizes
    df["combined_effect_size"] = np.sqrt(
        df["abs_dz_ss"].astype(float) * df["abs_dz_soth"].astype(float)
    )

    # Z-scores approximated from p-values and effect size sign
    df["dz_ss_z"] = df.apply(
        lambda row: _compute_z_score(row["p_ss"], row["dz_ss_mean"]), axis=1
    )
    df["dz_soth_z"] = df.apply(
        lambda row: _compute_z_score(row["p_soth"], row["dz_soth_mean"]), axis=1
    )

    # Combined z-score magnitude (Euclidean norm of individual z-scores)
    df["combined_z_score"] = np.sqrt(
        df["dz_ss_z"] ** 2 + df["dz_soth_z"] ** 2
    )

    # Statistical power indicator: ratio of combined_z_score^2 to itself + total sample size
    # Since total sample size per pair is unknown here, we approximate using combined_z_score
    df["statistical_power"] = df["combined_z_score"] ** 2 / (
        df["combined_z_score"] ** 2 + 1.0
    )

    # Differential effect size (septic shock minus sepsis)
    df["delta_effect_size"] = df["dz_soth_mean"] - df["dz_ss_mean"]
    df["abs_delta_effect_size"] = df["delta_effect_size"].abs()

    # Placeholder for baseline similarity; to be filled in downstream
    df["baseline_similarity"] = np.nan

    return df


def compute_baseline_similarity(
    df: pd.DataFrame, baseline_pair: Tuple[str, str]
) -> pd.Series:
    """Compute a simple similarity metric to a baseline gene pair.

    The similarity is based on effect size patterns: Euclidean
    similarity in the 2D space of ``dz_ss_mean`` and ``dz_soth_mean``
    relative to the baseline. The result is scaled to [0,1], where 1
    indicates identity and 0 indicates maximum distance observed in the
    dataset.

    Parameters
    ----------
    df : DataFrame
        Data containing ``dz_ss_mean`` and ``dz_soth_mean`` for each
        pair.
    baseline_pair : tuple
        Tuple of gene names (GeneAName, GeneBName) designating the
        positive control pair. The corresponding row must be present in
        the DataFrame; otherwise a runtime error is raised.

    Returns
    -------
    Series
        Similarity scores for each row in the DataFrame.
    """
    # Find the baseline row
    a, b = baseline_pair
    baseline_row = df[(df["GeneAName"] == a) & (df["GeneBName"] == b)]
    if baseline_row.empty:
        raise ValueError(
            f"Baseline pair {baseline_pair} not found in input data; cannot compute similarity"
        )
    base_effects = baseline_row.iloc[0][["dz_ss_mean", "dz_soth_mean"]].values.astype(float)

    # Compute Euclidean distance to baseline in effect size space
    effects = df[["dz_ss_mean", "dz_soth_mean"]].values.astype(float)
    distances = np.linalg.norm(effects - base_effects, axis=1)
    max_dist = distances.max() if distances.max() > 0 else 1.0
    similarity = 1.0 - distances / max_dist

    return pd.Series(similarity, index=df.index)