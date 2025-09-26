import numpy as np
import pandas as pd
from stats import bh_fdr

def apply_gates(met, th):
    ok = (
        met["q_ss"].le(th["q_max"]) &
        met["q_sothers"].le(th["q_max"]) &
        met["abs_dz_ss"].ge(th["abs_dz_min"]) &
        met["abs_dz_sothers"].ge(th["abs_dz_min"]) &
        met["kappa_ss"].ge(th["kappa_min"]) &
        met["kappa_sothers"].ge(th["kappa_min"])
    )
    return met.assign(pass_gates=ok)

def composite_score(met):
    # tune as needed
    s = (
        met["abs_dz_ss"] * np.log10(1/met["q_ss"].clip(1e-300) + 1) *
        met["kappa_ss"].fillna(0) * np.sqrt(met["auroc"].fillna(0.5))
    )
    return s

def rank(met):
    met = met.copy()
    met["score"] = composite_score(met)
    return met.sort_values("score", ascending=False)
