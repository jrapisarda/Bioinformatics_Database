import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_feature_engineering():
    module_path = Path(__file__).resolve().parents[2] / "ML" / "gene_pair_agent" / "feature_engineering.py"
    spec = importlib.util.spec_from_file_location("gene_pair_feature_engineering", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FeatureEngineering


def test_create_derived_features_column_sets():
    FeatureEngineering = load_feature_engineering()
    fe = FeatureEngineering()

    data = pd.DataFrame({
        "dz_ss_mean": np.array([1.0, -0.5]),
        "dz_soth_mean": np.array([0.5, 0.25]),
        "p_ss": np.array([0.01, 0.2]),
        "p_soth": np.array([0.05, 0.4]),
        "dz_ss_ci_low": np.array([0.8, -0.7]),
        "dz_ss_ci_high": np.array([1.2, -0.3]),
        "dz_soth_ci_low": np.array([0.4, 0.1]),
        "dz_soth_ci_high": np.array([0.6, 0.3]),
        "dz_ss_I2": np.array([0.2, 0.6]),
        "dz_soth_I2": np.array([0.1, 0.2]),
        "dz_ss_z": np.array([1.5, -2.0]),
        "dz_soth_z": np.array([0.5, 1.0]),
        "n_studies_ss": np.array([10, 5]),
        "n_studies_soth": np.array([8, 4]),
        "q_ss": np.array([0.02, 0.4]),
        "q_soth": np.array([0.03, 0.5]),
    })

    result = fe.create_derived_features(data)

    derived_columns = {
        "effect_size_ratio",
        "effect_size_diff",
        "effect_size_sum",
        "effect_size_product",
        "log_p_ss",
        "log_p_soth",
        "p_composite",
        "p_harmonic_mean",
        "ci_ss_width",
        "ci_ss_center",
        "ci_ss_relative_width",
        "ci_soth_width",
        "ci_soth_center",
        "ci_soth_relative_width",
        "i2_ratio",
        "i2_max",
        "i2_min",
        "i2_mean",
        "z_composite",
        "z_ratio",
        "z_sum",
        "z_product",
        "total_studies",
        "study_ratio",
        "study_log_ratio",
        "log_q_ss",
        "log_q_soth",
        "q_composite",
        "q_max",
        "q_min",
    }

    assert derived_columns.issubset(result.columns)

    numeric_cols = [
        col for col in data.columns.union(pd.Index(derived_columns))
        if col not in {"pair_id", "GeneAKey", "GeneBKey"}
    ]
    expected_abs_columns = {f"{col}_abs" for col in numeric_cols}

    assert expected_abs_columns.issubset(result.columns)

    # Ensure the original columns remain unchanged
    for column in data.columns:
        pd.testing.assert_series_equal(result[column], data[column])
