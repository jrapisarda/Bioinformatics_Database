import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_chart_generator():
    module_path = Path(__file__).resolve().parents[2] / "ML" / "visualization" / "chart_generator.py"
    spec = importlib.util.spec_from_file_location("chart_generator", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ChartGenerator


def test_create_boxplot_returns_empty_chart_for_non_finite_values():
    ChartGenerator = load_chart_generator()
    generator = ChartGenerator()

    data = pd.DataFrame({
        "dz_ss_mean": [np.nan, np.inf, -np.inf],
        "dz_soth_mean": [np.nan, np.inf, -np.inf],
    })

    result = generator.create_boxplot(data, results={})

    assert result["layout"]["annotations"][0]["text"] == "No effect size data available"


def test_create_scatter_plot_returns_empty_chart_for_non_finite_pairs():
    ChartGenerator = load_chart_generator()
    generator = ChartGenerator()

    data = pd.DataFrame({
        "dz_ss_mean": [np.nan, np.inf],
        "dz_soth_mean": [np.nan, -np.inf],
        "GeneAName": ["GeneA1", "GeneA2"],
        "GeneBName": ["GeneB1", "GeneB2"],
    })

    result = generator.create_scatter_plot(data, results={})

    assert result["layout"]["annotations"][0]["text"] == "No paired effect sizes"
