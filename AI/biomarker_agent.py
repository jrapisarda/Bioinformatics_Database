import json, math
import numpy as np
import pandas as pd
from pathlib import Path
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from scipy.stats import norm

# ------------------ Config helpers ------------------

def load_cfg(path: str) -> dict:
    cfg = json.loads(Path(path).read_text())
    # sensible defaults
    cfg.setdefault("random_seed", 42)
    cfg.setdefault("thresholds", {"q_max": 0.05, "abs_dz_min": 0.40, "kappa_min": 0.7})
    return cfg

def make_engine(cfg: dict):
    d = cfg["db"]
    if d.get("trusted_connection"):
        cs = (
            f"DRIVER={{{d['driver']}}};SERVER={d['server']};DATABASE={d['database']};"
            "Trusted_Connection=yes;TrustServerCertificate=yes"
        )
    else:
        cs = (
            f"DRIVER={{{d['driver']}}};SERVER={d['server']};DATABASE={d['database']};"
            f"UID={d['username']};PWD={d['password']};TrustServerCertificate=yes"
        )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(cs)}", future=True)

# ------------------ Stats utilities ------------------

def fisher_z(r):
    # Spearman r in [-1,1]; clamp to avoid inf
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = np.isfinite(p).sum()
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    q = np.full_like(p, np.nan, float)
    prev = 1.0
    rank = 0
    for idx in order:
        if not np.isfinite(p[idx]): continue
        rank += 1
        val = p[idx] * m / rank
        prev = min(prev, val)
        q[idx] = prev
    return q

def dl_random_effects(effects, ses):
    effects = np.asarray(effects, float); ses = np.asarray(ses, float)
    mask = np.isfinite(effects) & np.isfinite(ses) & (ses > 0)
    effects, ses = effects[mask], ses[mask]
    k = effects.size
    if k == 0:
        return dict(k=0, mean=np.nan, se=np.nan, ci=(np.nan, np.nan), Q=np.nan, I2=np.nan, tau2=np.nan, z=np.nan, p=np.nan)
    w = 1.0 / (ses ** 2)
    mu_fixed = np.sum(w * effects) / np.sum(w)
    Q = np.sum(w * (effects - mu_fixed) ** 2)
    df = k - 1
    C = np.sum(w) - (np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - df) / C) if df > 0 else 0.0
    w_re = 1.0 / (ses ** 2 + tau2)
    mu_re = np.sum(w_re * effects) / np.sum(w_re)
    se_re = (1.0 / np.sum(w_re)) ** 0.5
    zcrit = norm.ppf(0.975)
    ci = (mu_re - zcrit * se_re, mu_re + zcrit * se_re)
    I2 = max(0.0, (Q - df) / Q) * 100.0 if (Q > 0 and df > 0) else 0.0
    z = mu_re / se_re if se_re > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return dict(k=k, mean=mu_re, se=se_re, ci=ci, Q=Q, I2=I2, tau2=tau2, z=z, p=p)

def sign_agreement(deltas, pooled_mean):
    vals = np.asarray(deltas, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or not np.isfinite(pooled_mean) or pooled_mean == 0:
        return np.nan
    return float((np.sign(vals) == np.sign(pooled_mean)).mean())

# ------------------ Data access ------------------

VIEW_SQL = text("""
SELECT
  geneA, geneB,
  geneA_ss_sepsis, geneB_ss_sepsis,
  geneA_ss_ctrl,   geneB_ss_ctrl,
  geneA_ss_direction, geneB_ss_direction,
  illness_label,
  rho_spearman, p_value, q_value, n_samples,
  study_key,
  GeneAName, GeneBName, GeneAKey, GeneBKey
FROM dbo.vw_gene_DE_fact_corr_data
""")



def load_studywise(engine) -> pd.DataFrame:
    df = pd.read_sql(VIEW_SQL, engine)
    # stable numeric pair_id based on keys
    ga, gb = df["GeneAKey"].astype(int), df["GeneBKey"].astype(int)
    a_min = ga.where(ga <= gb, gb)
    b_max = gb.where(ga <= gb, ga)
    df["pair_id"] = a_min.astype(str) + "_" + b_max.astype(str)
    return df

# ------------------ Core computation ------------------

def compute_metrics(df: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """
    Build per-pair meta-analysis for:
      1) Shock (1) vs Sepsis (3) : Δz_ss   = z_shock - z_sepsis
         SE_ss = sqrt( 1/(n_shock-3) + 1/(n_sepsis-3) )
      2) Shock (1) vs Others     : Δz_soth = z_shock - z_others, where
         z_others = precision-weighted mean of (z_control, z_sepsis),
         var(z_others) = 1 / (w_ctrl + w_seps), with w = (n-3)
         SE_soth = sqrt( 1/(n_shock-3) + 1/(w_ctrl+w_seps) )
    """
    # Map illness_label ints to names we’ll use consistently
    inv = {"septic shock": "septic shock",
       "control":      "control",
       "sepsis":       "sepsis"}  # {1:'septic_shock', 2:'control', 3:'sepsis'}
    df = df.copy()
    df["group"] = df["illness_label"].map(inv)

    # ==== inside compute_metrics(), immediately after df["group"] = ... ====
    print("\n1️⃣ LABEL MAP SANITY CHECK")
    print("  Raw illness_label values :", sorted(df["illness_label"].dropna().unique()))
    print("  group col NaN count      :", df["group"].isna().sum())
    print("  head of illness_label → group:\n", df[["illness_label", "group"]].head())

    print("Unique illness_label raw values:", sorted(df["illness_label"].dropna().unique()))
    print("Mapping dict inv:", inv)
    print("Group col after map (head):\n", df[["illness_label", "group"]].head())
    print("NaN count in group:", df["group"].isna().sum())

    # compute Fisher z and keep counts per study/group
    df["z"] = df["rho_spearman"].apply(fisher_z)
    df["n_eff"] = df["n_samples"].apply(lambda n: np.nan if (pd.isna(n) or n <= 3) else float(n - 3))

    # Pivot to per-study rows for each pair
    pvt = df.pivot_table(index=["pair_id", "study_key"],
                         columns="group",
                         values=["z", "n_eff"],
                         aggfunc="first")
    
    # ==== right after pvt = df.pivot_table(...) ====
    print("\n2️⃣ PIVOT RESULT")
    print("  Pivot shape (study×pair, groups×vars) :", pvt.shape)
    print("  Pivot column index :", pvt.columns.tolist())
    print("  Missing-value rate per column:\n", pvt.isna().mean().round(2))

    # Helper to get column safely
    def col(mat, root, grp):
        try:
            return mat[(root, grp)]
        except KeyError:
            return pd.Series(index=mat.index, dtype="float64")

    z_shock   = col(pvt, "z", "septic shock")
    z_sepsis  = col(pvt, "z", "sepsis")
    z_ctrl    = col(pvt, "z", "control")
    ne_shock  = col(pvt, "n_eff", "septic shock")
    ne_sepsis = col(pvt, "n_eff", "sepsis")
    ne_ctrl   = col(pvt, "n_eff", "control")

    # Per-study deltas & SEs
    dz_ss   = z_shock - z_sepsis
    se_ss   = np.sqrt(1.0 / ne_shock + 1.0 / ne_sepsis)

    w_ctrl  = ne_ctrl.fillna(0.0)
    w_seps  = ne_sepsis.fillna(0.0)
    z_others = (w_ctrl * z_ctrl.fillna(0.0) + w_seps * z_sepsis.fillna(0.0)) / (w_ctrl + w_seps).replace(0, np.nan)
    var_others = 1.0 / (w_ctrl + w_seps)
    dz_soth = z_shock - z_others
    se_soth = np.sqrt(1.0 / ne_shock + var_others)

    # Drop rows with any NaNs in effect or SE
    per_study = pd.DataFrame({
        "pair_id": dz_ss.index.get_level_values("pair_id"),
        "study_key": dz_ss.index.get_level_values("study_key"),
        "dz_ss": dz_ss.values, "se_ss": se_ss.values,
        "dz_soth": dz_soth.values, "se_soth": se_soth.values
    })

    # ==== right after per_study = pd.DataFrame({...}) ====
    print("\n3️⃣ PER-STUDY DELTAS")
    print("  per_study shape :", per_study.shape)
    print("  per_study head:\n", per_study.head())
    print("  NaN per column  :\n", per_study.isna().sum())

    print("Before dropna:")
    print("  dz_ss NaN   :", dz_ss.isna().sum())
    print("  se_ss NaN   :", se_ss.isna().sum())
    print("  dz_soth NaN :", dz_soth.isna().sum())
    print("  se_soth NaN :", se_soth.isna().sum())
    # ---- 3️⃣-bis  “why is it NaN?” ----
    print("\n3️⃣-bis  MISSING-COMPONENT CHECK")
    for col, vec in (("z_shock", z_shock), ("z_sepsis", z_sepsis), ("z_ctrl", z_ctrl),
                 ("ne_shock", ne_shock), ("ne_sepsis", ne_sepsis), ("ne_ctrl", ne_ctrl)):
        print(f"  {col:10}  NaN : {vec.isna().sum():5}   ({vec.isna().mean():.1%})")


    per_study = per_study.replace([np.inf, -np.inf], np.nan).dropna(subset=["dz_ss", "se_ss", "dz_soth", "se_soth"], how="all")

    # Aggregate by pair_id via random-effects meta
    rows = []
    for pid, g in per_study.groupby("pair_id"):
        eff_ss, ses_ss     = g["dz_ss"].values,   g["se_ss"].values
        eff_soth, ses_soth = g["dz_soth"].values, g["se_soth"].values

        re_ss   = dl_random_effects(eff_ss,   ses_ss)   if eff_ss.size   >= 1 else {k: np.nan for k in ["mean","se","ci","Q","I2","z","p","k"]}
        re_soth = dl_random_effects(eff_soth, ses_soth) if eff_soth.size >= 1 else {k: np.nan for k in ["mean","se","ci","Q","I2","z","p","k"]}

        kappa_ss    = sign_agreement(eff_ss, re_ss.get("mean", np.nan))
        kappa_soth  = sign_agreement(eff_soth, re_soth.get("mean", np.nan))

        rows.append(dict(
            pair_id=pid,
            n_studies_ss = int(np.unique(g.loc[~np.isnan(eff_ss), "study_key"]).size),
            n_studies_soth = int(np.unique(g.loc[~np.isnan(eff_soth), "study_key"]).size),

            dz_ss_mean = re_ss.get("mean", np.nan),
            dz_ss_se   = re_ss.get("se",   np.nan),
            dz_ss_ci_low  = (re_ss.get("ci") or (np.nan, np.nan))[0],
            dz_ss_ci_high = (re_ss.get("ci") or (np.nan, np.nan))[1],
            dz_ss_Q = re_ss.get("Q", np.nan),
            dz_ss_I2 = re_ss.get("I2", np.nan),
            dz_ss_z  = re_ss.get("z", np.nan),
            p_ss     = re_ss.get("p", np.nan),

            dz_soth_mean = re_soth.get("mean", np.nan),
            dz_soth_se   = re_soth.get("se",   np.nan),
            dz_soth_ci_low  = (re_soth.get("ci") or (np.nan, np.nan))[0],
            dz_soth_ci_high = (re_soth.get("ci") or (np.nan, np.nan))[1],
            dz_soth_Q = re_soth.get("Q", np.nan),
            dz_soth_I2 = re_soth.get("I2", np.nan),
            dz_soth_z  = re_soth.get("z", np.nan),
            p_soth     = re_soth.get("p", np.nan),

            kappa_ss = kappa_ss,
            kappa_soth = kappa_soth,
            abs_dz_ss = abs(re_ss.get("mean", np.nan)),
            abs_dz_soth = abs(re_soth.get("mean", np.nan)),
        ))

    met = pd.DataFrame(rows)
    # FDR across all pairs
    if "p_ss" in met:
        met["q_ss"] = bh_fdr(met["p_ss"].values)
    if "p_soth" in met:
        met["q_soth"] = bh_fdr(met["p_soth"].values)
    
    print("Rows collected:", len(rows))
    print("DataFrame shape:", met.shape)
    print("Columns:", met.columns.tolist())

    # Rank: stronger Δz_ss with low q_ss first; tiebreaker Δz_soth
    # (You can plug in your composite score here)
    met["rank_score"] = (
        met["abs_dz_ss"].fillna(0) * np.log10(1.0 / met["q_ss"].clip(lower=1e-300, upper=1.0) + 1.0) *
        met["kappa_ss"].fillna(0)
    ) + 0.25 * (
        met["abs_dz_soth"].fillna(0) * np.log10(1.0 / met["q_soth"].clip(lower=1e-300, upper=1.0) + 1.0) *
        met["kappa_soth"].fillna(0)
    )
    met = met.sort_values("rank_score", ascending=False).reset_index(drop=True)
    return met

# ------------------ Orchestrator ------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON with db creds and label map")
    ap.add_argument("--out", default="pair_metrics_agent.csv")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    engine = make_engine(cfg)

    # Required label map (you provided): Control=2, Septic Shock=1, Sepsis=3
    labels = cfg["labels"]  # e.g., {"control": 2, "septic_shock": 1, "sepsis": 3}

    # 1) Load studywise correlations from your view
    df = load_studywise(engine)

    # 2) Compute pooled metrics for (Shock vs Sepsis) and (Shock vs Others)
    met = compute_metrics(df, labels)

    # 3) (Optional) join gene names if you want them in the CSV
    name_cols = df.groupby("pair_id")[["GeneAName","GeneBName","GeneAKey","GeneBKey"]].first().reset_index()
    met = met.merge(name_cols, on="pair_id", how="left")

    # 4) Save
    met.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  rows={len(met)}")

if __name__ == "__main__":
    main()
