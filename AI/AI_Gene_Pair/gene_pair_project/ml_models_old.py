"""
Machine learning models for unsupervised analysis of gene pairs.

This module implements a lightweight ensemble of unsupervised algorithms
to discover unusual gene pair relationships. It combines an
IsolationForest for anomaly detection with a DBSCAN clustering model to
identify dense regions of similar pairs. The resulting scores can be
used as part of the final recommendation ranking.

The models expect a set of numeric feature columns derived from raw
meta‐analysis statistics (see :mod:`gene_pair_project.data_processing`).
All features are standardised to zero mean and unit variance before
modelling to ensure robustness to scale differences. Negative values
present no issue as long as the data are centred and scaled.

Note that the scores produced here are relative measures; they do not
represent probabilities. A lower IsolationForest score indicates a more
anomalous pair. Cluster labels of ``-1`` from DBSCAN denote noise
points.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN


def compute_ml_scores(
    df_features: pd.DataFrame,
    feature_columns: List[str] | None = None,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute unsupervised scores using IsolationForest and DBSCAN.

    Parameters
    ----------
    df_features : DataFrame
        DataFrame containing engineered numeric features. Missing
        values will be filled with zeros prior to scaling.
    feature_columns : list of str, optional
        Names of columns to use for modelling. If ``None``, a
        reasonable default subset of engineered features is used.
    random_state : int, default=42
        Seed for the IsolationForest to ensure reproducibility.

    Returns
    -------
    tuple of Series
        A tuple ``(ml_score, is_outlier, cluster_label)`` where
        ``ml_score`` is a normalised score in [0,1] (higher means more
        interesting), ``is_outlier`` is a boolean flag indicating
        whether the pair was marked as an anomaly by IsolationForest,
        and ``cluster_label`` is the cluster id assigned by DBSCAN
        (``-1`` for noise).
    """
    if feature_columns is None:
        # Default subset of features considered informative
        candidate_cols = [
            "combined_p_value",
            "combined_effect_size",
            "combined_z_score",
            "abs_delta_effect_size",
            "statistical_power",
        ]
        feature_columns = [c for c in candidate_cols if c in df_features.columns]
        if not feature_columns:
            raise ValueError(
                "No valid numeric feature columns were found in the DataFrame."
            )

    X = df_features[feature_columns].astype(float).fillna(0.0).values
    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest for anomaly detection
    iso = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=random_state,
    )
    iso.fit(X_scaled)
    anomaly_scores = -iso.decision_function(X_scaled)  # higher means more anomalous
    is_outlier = pd.Series(iso.predict(X_scaled) == -1, index=df_features.index)

    # DBSCAN clustering to identify dense groups
    # eps chosen heuristically; min_samples derived from total count
    eps = 0.5
    min_samples = max(5, int(0.005 * len(df_features)))
    db = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = pd.Series(db.fit_predict(X_scaled), index=df_features.index)

    # Estimate cluster density: inverse of cluster size; noise points get zero density
    label_counts = cluster_labels.value_counts().to_dict()
    cluster_density = cluster_labels.map(
        lambda lbl: 0.0 if lbl == -1 else 1.0 / label_counts.get(lbl, 1)
    )

    # Normalise anomaly scores to [0,1]
    if anomaly_scores.max() - anomaly_scores.min() > 0:
        anomaly_norm = (anomaly_scores - anomaly_scores.min()) / (
            anomaly_scores.max() - anomaly_scores.min()
        )
    else:
        anomaly_norm = np.zeros_like(anomaly_scores)

    # Combined ML score: emphasise anomalies (interesting outliers) and cluster density
    combined_ml = 0.7 * anomaly_norm + 0.3 * cluster_density.values

    # Clip to [0,1]
    combined_ml = np.clip(combined_ml, 0, 1)

    ml_score_series = pd.Series(combined_ml, index=df_features.index)

    return ml_score_series, is_outlier, cluster_labels