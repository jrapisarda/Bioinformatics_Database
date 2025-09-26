"""
Gene Pair Analysis Project
==========================

This package provides functionality to load and process gene pair
meta‐analysis results, apply a configurable rule engine, compute
ensemble machine learning scores, and generate interpretable
recommendations. It also includes a simple Flask web application for
uploading input data, configuring ranking rules, and exploring
recommendations through interactive visualisations.

Modules
-------

* `data_processing` – utilities for loading and validating input data
  from CSV, Excel or JSON files and engineering features for
  downstream analysis.
* `rules` – default rule definitions and helper functions for
  loading, saving and modifying rules.
* `rule_engine` – functions to safely evaluate rule conditions on
  per‐pair data and compute aggregate rule scores.
* `ml_models` – wrappers around unsupervised algorithms used to
  produce anomaly and similarity scores.
* `recommendation` – functions for assembling final rankings from
  raw statistical measures, machine learning scores and rule
  evaluations.
* `app` – a minimal Flask application exposing a web interface for
  uploading data, configuring rules and visualising results.

This project is intended to be configured and extended by end users
through the web interface or programmatic API. Default behaviour is
driven by the requirements defined in the accompanying
`gene-pair-ml-requirements.md` document.
"""

from . import data_processing  # noqa: F401
from . import rules  # noqa: F401
from . import rule_engine  # noqa: F401
from . import ml_models  # noqa: F401
from . import recommendation  # noqa: F401