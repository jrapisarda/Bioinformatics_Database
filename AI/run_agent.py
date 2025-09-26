import json, numpy as np, pandas as pd
from collections import defaultdict
from sqlalchemy import text

from db import make_engine
from data_access import load_studywise_corr
from stats import fisher_z, welch, bh_fdr, sign_kappa_like, dl_random_effects
from ranker import apply_gates, rank

def compute_pair_metrics(df, labels, cfg):
    # df columns: study_key, pair_id, group, r
    # pivot to per-study rows with cols per group: control, sepsis, septic shock
    pvt = df.pivot_table(index=["pair_id","study_key"], columns="group", values="r", aggfunc="first").reset_index()
    # Fisher z per cell
    for col in ["control","sepsis","septic shock"]:
        if col in pvt: pvt[col] = pvt[col].apply(fisher_z)

    met_rows = []
    for pid, g in pvt.groupby("pair_id"):
        # Arrays of per-study Fisher z
        arr_shock  = g["septic shock"].dropna().values if "septic shock" in g else np.array([])
        arr_sepsis = g["sepsis"].dropna().values       if "sepsis" in g else np.array([])
        arr_ctrl   = g["control"].dropna().values      if "control" in g else np.array([])

        # Study-level deltas
        dz_ss = None; dz_sothers = None
        if "septic shock" in g and "sepsis" in g:
            dz_ss = (g["septic shock"] - g["sepsis"]).dropna().values
        if "septic shock" in g:
            others_cols = [c for c in ["control","sepsis"] if c in g]
            if others_cols:
                others_mean = g[others_cols].mean(axis=1)
                dz_sothers = (g["septic shock"] - others_mean).dropna().values

        # Means & kappas
        mean_dz_ss      = float(np.nanmean(dz_ss)) if dz_ss is not None and dz_ss.size else np.nan
        mean_dz_sothers = float(np.nanmean(dz_sothers)) if dz_sothers is not None and dz_sothers.size else np.nan
        kappa_ss        = sign_kappa_like(dz_ss, mean_dz_ss)
        kappa_sothers   = sign_kappa_like(dz_sothers, mean_dz_sothers)

        # Welch tests on per-study z
        t_ss, p_ss = welch(arr_shock, arr_sepsis) if (arr_shock.size>=2 and arr_sepsis.size>=2) else (np.nan, np.nan)
        arr_others = np.concatenate([a for a in (arr_ctrl, arr_sepsis) if a.size]) if (arr_ctrl.size+arr_sepsis.size)>0 else np.array([])
        t_sothers, p_sothers = welch(arr_shock, arr_others) if (arr_shock.size>=2 and arr_others.size>=2) else (np.nan, np.nan)

        # Optional random-effects on study deltas (rough SE proxy: sd/√k)
        re_ss = re_sothers = dict(mean=np.nan, ci=(np.nan,np.nan), I2=np.nan, Q=np.nan)
        if cfg.get("meta",{}).get("random_effects", True):
            if dz_ss is not None and dz_ss.size>=2:
                sd = np.std(dz_ss, ddof=1); se = np.full_like(dz_ss, sd/np.sqrt(dz_ss.size))
                d = dl_random_effects(dz_ss, se); re_ss = dict(mean=d["mean"], ci=d["ci"], I2=d["I2"], Q=d["Q"])
            if dz_sothers is not None and dz_sothers.size>=2:
                sd = np.std(dz_sothers, ddof=1); se = np.full_like(dz_sothers, sd/np.sqrt(dz_sothers.size))
                d = dl_random_effects(dz_sothers, se); re_sothers = dict(mean=d["mean"], ci=d["ci"], I2=d["I2"], Q=d["Q"])

        met_rows.append(dict(
            pair_id=pid,
            n_studies=int(g["study_key"].nunique()),
            mean_dz_ss=mean_dz_ss,  abs_dz_ss=abs(mean_dz_ss),
            mean_dz_sothers=mean_dz_sothers, abs_dz_sothers=abs(mean_dz_sothers),
            t_ss=t_ss, p_ss=p_ss, t_sothers=t_sothers, p_sothers=p_sothers,
            kappa_ss=kappa_ss, kappa_sothers=kappa_sothers,
            dz_ss_ci_low=re_ss["ci"][0], dz_ss_ci_high=re_ss["ci"][1], dz_ss_I2=re_ss["I2"], dz_ss_Q=re_ss["Q"],
            dz_sothers_ci_low=re_sothers["ci"][0], dz_sothers_ci_high=re_sothers["ci"][1], dz_sothers_I2=re_sothers["I2"], dz_sothers_Q=re_sothers["Q"]
        ))

    met = pd.DataFrame(met_rows)
    # FDR
    for pcol, qcol in [("p_ss","q_ss"), ("p_sothers","q_sothers")]:
        if pcol in met:
            met[qcol] = bh_fdr(met[pcol].values)
    return met

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    rng = np.random.default_rng(int(cfg.get("random_seed", 42)))

    engine = make_engine(cfg)
    labels = cfg["labels"]  # {control:2,septic_shock:1,sepsis:3}

    # 1) Load studywise correlations (fast; leverages your view)
    df = load_studywise_corr(engine, labels)

    # 2) Compute pair metrics for: Shock vs Sepsis; Shock vs Others
    met = compute_pair_metrics(df, labels, cfg)

    # 3) Gate + rank
    met = apply_gates(met, cfg["thresholds"])
    met = rank(met)

    # 4) Persist results (table or CSV); simple CSV here
    met.to_csv("pair_metrics_agent.csv", index=False)
    print("Wrote pair_metrics_agent.csv with", len(met), "rows.")

if __name__ == "__main__":
    main()
