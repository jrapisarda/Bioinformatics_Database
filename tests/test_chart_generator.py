import base64

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AI.ML.visualization.chart_generator import ChartGenerator


def _decode_numeric_array(array_payload):
    """Decode Plotly's binary array payload into a numpy array."""
    if isinstance(array_payload, dict) and 'bdata' in array_payload:
        return np.frombuffer(base64.b64decode(array_payload['bdata']), dtype=np.float64)
    return np.asarray(array_payload, dtype=float)


def test_create_boxplot_returns_empty_chart_when_no_effect_sizes():
    data = pd.DataFrame({
        'dz_ss_mean': [np.nan, np.nan],
        'dz_soth_mean': [np.nan, np.nan],
    })

    chart = ChartGenerator().create_boxplot(data, {})

    assert chart['layout']['title']['text'] == 'Chart Unavailable'
    assert chart['layout']['annotations'][0]['text'] == 'No effect size data available'


def test_create_boxplot_discard_empty_series():
    data = pd.DataFrame({
        'dz_ss_mean': [np.nan, np.inf, 0.5],
        'dz_soth_mean': [np.nan, -np.inf, np.nan],
    })

    chart = ChartGenerator().create_boxplot(data, {})

    assert len(chart['data']) == 1
    box_trace = chart['data'][0]
    decoded = _decode_numeric_array(box_trace['y'])
    assert np.allclose(decoded, np.array([0.5]))


def test_create_scatter_plot_filters_non_finite_values():
    data = pd.DataFrame({
        'dz_ss_mean': [0.1, np.nan, 0.3],
        'dz_soth_mean': [0.2, 0.4, np.nan],
        'p_ss': [0.01, 0.2, 0.2],
        'p_soth': [0.001, 0.2, 0.2],
        'GeneAName': ['A1', 'A2', 'A3'],
        'GeneBName': ['B1', 'B2', 'B3'],
    })

    chart = ChartGenerator().create_scatter_plot(data, {})

    scatter = chart['data'][0]
    x_values = _decode_numeric_array(scatter['x'])
    y_values = _decode_numeric_array(scatter['y'])

    assert len(x_values) == 1
    assert len(y_values) == 1
    np.testing.assert_allclose(x_values, np.array([0.1]))
    np.testing.assert_allclose(y_values, np.array([0.2]))
    assert scatter['text'] == ['Gene A: A1<br>Gene B: B1']


def test_create_scatter_plot_returns_empty_chart_with_no_finite_values():
    data = pd.DataFrame({
        'dz_ss_mean': [np.nan, np.inf],
        'dz_soth_mean': [np.nan, -np.inf],
        'GeneAName': ['A1', 'A2'],
        'GeneBName': ['B1', 'B2'],
    })

    chart = ChartGenerator().create_scatter_plot(data, {})

    assert chart['layout']['title']['text'] == 'Chart Unavailable'
    assert chart['layout']['annotations'][0]['text'] == 'No paired effect sizes'
