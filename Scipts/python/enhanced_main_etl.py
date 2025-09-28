"""Enhanced ETL utilities for measurement technology inference."""
from __future__ import annotations

import re
from typing import Optional


def _normalise_descriptor(value: Optional[str]) -> str:
    """Normalise platform or study technology descriptors for comparison.

    The normalisation routine lowercases the input, strips leading/trailing
    whitespace, and collapses whitespace/underscore/dash groups into a single
    space so that variants such as ``"RNA_SEQ"`` and ``"rna-seq"`` can be
    compared consistently.
    """
    if not value:
        return ""

    # Replace underscores and dashes with spaces before collapsing repeated
    # whitespace to a single space. This keeps terms tokenised without
    # punctuation noise.
    cleaned = re.sub(r"[\-_]+", " ", value.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _infer_measurement_technology(
    study_technology: Optional[str], platform_name: Optional[str]
) -> str:
    """Infer the measurement technology for a study.

    The study-level technology hint is normalised and preferred over platform
    information. This allows descriptors such as ``"RNA_SEQ"`` or
    ``"micro array"`` to be interpreted correctly before falling back to the
    platform name heuristics.
    """

    normalised_study = _normalise_descriptor(study_technology)
    if normalised_study:
        compact_study = normalised_study.replace(" ", "")
        tokens = normalised_study.split()

        # Check for microarray indicators first to avoid accidental matches on
        # the "rna" substring within "microarray".
        if "microarray" in compact_study:
            return "MICROARRAY"

        # Accept the common RNA-Seq variations (rna seq, rnaseq, rna sequencing).
        if (
            "rnaseq" in compact_study
            or "rna seq" in normalised_study
            or ("rna" in tokens and ("seq" in tokens or "sequencing" in tokens))
        ):
            return "RNA-SEQ"

    # Fallback: infer from the platform name when the study metadata is not
    # decisive.
    normalised_platform = _normalise_descriptor(platform_name)
    if normalised_platform:
        compact_platform = normalised_platform.replace(" ", "")
        platform_tokens = normalised_platform.split()

        if "microarray" in compact_platform or "array" in platform_tokens:
            return "MICROARRAY"

        if (
            "rnaseq" in compact_platform
            or "rna seq" in normalised_platform
            or ("rna" in platform_tokens and ("seq" in platform_tokens or "sequencing" in platform_tokens))
        ):
            return "RNA-SEQ"

    return "OTHER"
