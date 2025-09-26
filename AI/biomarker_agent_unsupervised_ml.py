import json, math
import numpy as np
import pandas as pd
from pathlib import Path
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from scipy.stats import norm
import logging

# Set up basic logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
# Additional imports for logging and unsupervised learning

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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
FROM dbo.vw_gene_DE_fact_corr_data WHERE geneA_ss_direction <> geneB_ss_direction
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

    logger.debug("Label map applied. Unique illness labels: %s", sorted(df["illness_label"].dropna().unique()))
    logger.debug("Group column NaN count: %s", df["group"].isna().sum())
    logger.debug("Illness label → group head:\n%s", df[["illness_label", "group"]].head())

    # compute Fisher z and keep counts per study/group
    df["z"] = df["rho_spearman"].apply(fisher_z)
    df["n_eff"] = df["n_samples"].apply(lambda n: np.nan if (pd.isna(n) or n <= 3) else float(n - 3))

    # Pivot to per-study rows for each pair
    pvt = df.pivot_table(index=["pair_id", "study_key"],
                         columns="group",
                         values=["z", "n_eff"],
                         aggfunc="first")
    
    logger.debug("Pivot result shape: %s", pvt.shape)
    logger.debug("Pivot columns: %s", pvt.columns.tolist())
    logger.debug("Pivot missing-value rate:\n%s", pvt.isna().mean().round(2))

    groups = ["septic shock", "sepsis", "control"]
    for root in ("z", "n_eff"):
        for grp in groups:
            if (root, grp) not in pvt.columns:
                pvt[(root, grp)] = np.nan
    pvt = pvt.sort_index(axis=1)

    # Apply option C: if exactly one group is missing but the other two exist, impute with zeros
    z_frame = pvt["z"].copy()
    available_counts = z_frame.notna().sum(axis=1)
    missing_counts = z_frame.isna().sum(axis=1)
    mask_impute = (available_counts >= 2) & (missing_counts == 1)
    impute_counter = {grp: 0 for grp in groups}
    for idx in z_frame[mask_impute].index:
        missing_grp = z_frame.columns[z_frame.loc[idx].isna()][0]
        pvt.loc[idx, ("z", missing_grp)] = 0.0
        pvt.loc[idx, ("n_eff", missing_grp)] = 0.0
        impute_counter[missing_grp] += 1
    if any(impute_counter.values()):
        logger.debug("Imputed missing groups with zeros: %s", impute_counter)
    else:
        logger.debug("No rows required group-level imputation")

    # Helper to get column safely and fill missing with zeros
    def col(mat, root, grp):
        try:
            series = mat[(root, grp)]
        except KeyError:
            series = pd.Series(index=mat.index, dtype="float64")
        return series.fillna(0.0)

    z_shock   = col(pvt, "z", "septic shock")
    z_sepsis  = col(pvt, "z", "sepsis")
    z_ctrl    = col(pvt, "z", "control")
    ne_shock  = col(pvt, "n_eff", "septic shock")
    ne_sepsis = col(pvt, "n_eff", "sepsis")
    ne_ctrl   = col(pvt, "n_eff", "control")

    def safe_inverse(series: pd.Series) -> pd.Series:
        values = series.to_numpy(dtype=float, copy=True)
        result = np.zeros_like(values)
        valid = np.isfinite(values) & (values > 0)
        result[valid] = 1.0 / values[valid]
        return pd.Series(result, index=series.index)

    # Per-study deltas & SEs
    dz_ss   = z_shock - z_sepsis
    inv_shock = safe_inverse(ne_shock)
    inv_sepsis = safe_inverse(ne_sepsis)
    se_ss   = np.sqrt(inv_shock + inv_sepsis)

    w_ctrl  = ne_ctrl
    w_seps  = ne_sepsis
    numerator = (w_ctrl * z_ctrl) + (w_seps * z_sepsis)
    total_w = w_ctrl + w_seps
    total_w_nonzero = total_w.replace(0, np.nan)
    z_others = numerator.divide(total_w_nonzero).fillna(0.0)
    var_others = (1.0 / total_w_nonzero).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    se_soth = np.sqrt(inv_shock + var_others)
    dz_soth = z_shock - z_others

    # Drop rows with any NaNs in effect or SE
    per_study = pd.DataFrame({
        "pair_id": dz_ss.index.get_level_values("pair_id"),
        "study_key": dz_ss.index.get_level_values("study_key"),
        "dz_ss": dz_ss.values, "se_ss": se_ss.values,
        "dz_soth": dz_soth.values, "se_soth": se_soth.values
    })

    logger.debug("Per-study deltas shape: %s", per_study.shape)
    logger.debug("Per-study head:\n%s", per_study.head())
    logger.debug("Per-study NaN counts:\n%s", per_study.isna().sum())

    logger.debug(
        "NaNs before dropna - dz_ss: %s, se_ss: %s, dz_soth: %s, se_soth: %s",
        dz_ss.isna().sum(),
        se_ss.isna().sum(),
        dz_soth.isna().sum(),
        se_soth.isna().sum(),
    )
    for col, vec in (
        ("z_shock", z_shock),
        ("z_sepsis", z_sepsis),
        ("z_ctrl", z_ctrl),
        ("ne_shock", ne_shock),
        ("ne_sepsis", ne_sepsis),
        ("ne_ctrl", ne_ctrl),
    ):
        logger.debug("Missing count for %s: %s (%.1f%%)", col, vec.isna().sum(), vec.isna().mean() * 100)


    before_drop = len(per_study)
    per_study = per_study.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["dz_ss", "se_ss", "dz_soth", "se_soth"], how="all"
    )
    logger.debug(
        "Per-study rows retained after drop: %s (dropped %s)",
        len(per_study),
        before_drop - len(per_study),
    )

    # Aggregate by pair_id via random-effects meta
    rows = []
    # Precompute group-level meta-analysis metrics for each pair
    # We'll compute weighted correlation (r), standard error on r, heterogeneity (I2), z-statistic and p-value
    # for each illness group (control, sepsis, septic shock). We also derive differential correlation
    # statistics (delta r and associated z/p values) and combined composite metrics.
    # Create a mapping of valid groups for clarity
    valid_groups = ["control", "sepsis", "septic shock"]

    for pid, g in per_study.groupby("pair_id"):
        # Random effects meta for shock vs sepsis (dz_ss) and shock vs others (dz_soth)
        eff_ss, ses_ss     = g["dz_ss"].values,   g["se_ss"].values
        eff_soth, ses_soth = g["dz_soth"].values, g["se_soth"].values

        re_ss   = dl_random_effects(eff_ss,   ses_ss)   if eff_ss.size   >= 1 else {k: np.nan for k in ["mean","se","ci","Q","I2","z","p","k"]}
        re_soth = dl_random_effects(eff_soth, ses_soth) if eff_soth.size >= 1 else {k: np.nan for k in ["mean","se","ci","Q","I2","z","p","k"]}

        kappa_ss    = sign_agreement(eff_ss, re_ss.get("mean", np.nan))
        kappa_soth  = sign_agreement(eff_soth, re_soth.get("mean", np.nan))

        # Base row with existing metrics
        row_dict = dict(
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
        )

        # Compute group-level meta-analysis metrics using original df
        # Filter original df for current pair_id
        pair_df = df[df["pair_id"] == pid]
        # Initialize storage for weighted correlations and other statistics
        r_means = {}
        se_r_vals = {}
        I2_vals = {}
        z_stats = {}
        p_vals = {}
        sample_counts = {}

        for grp in valid_groups:
            grp_data = pair_df[pair_df["group"] == grp]
            # z values and effective sample sizes
            z_vals = grp_data["z"].dropna().values
            # n_eff already computed on df as n_samples - 3; ensure positive
            n_eff_vals = grp_data["n_eff"].dropna().values
            # standard error on z for each study
            se_z_vals = np.where(n_eff_vals > 0, 1.0 / np.sqrt(n_eff_vals), np.nan)
            if z_vals.size >= 1 and se_z_vals.size >= 1:
                # run random-effects meta-analysis on z-scale
                meta = dl_random_effects(z_vals, se_z_vals)
                m = meta.get("mean", np.nan)
                se_z = meta.get("se", np.nan)
                I2 = meta.get("I2", np.nan)
                z_stat = meta.get("z", np.nan)
                p_val = meta.get("p", np.nan)
                # back-transform to r
                r_mean = math.tanh(m) if (m is not None and np.isfinite(m)) else np.nan
                # standard error of r using delta method: se_r = se_z * (1 - r_mean**2)
                se_r = se_z * (1.0 - r_mean**2) if (se_z is not None and np.isfinite(se_z) and np.isfinite(r_mean)) else np.nan
                # store
                r_means[grp] = r_mean
                se_r_vals[grp] = se_r
                I2_vals[grp] = I2
                z_stats[grp] = z_stat
                p_vals[grp] = p_val
                # total samples for power score calculation
                sample_counts[grp] = grp_data["n_samples"].dropna().sum()
            else:
                r_means[grp] = np.nan
                se_r_vals[grp] = np.nan
                I2_vals[grp] = np.nan
                z_stats[grp] = np.nan
                p_vals[grp] = np.nan
                sample_counts[grp] = grp_data["n_samples"].dropna().sum()

        # Add group-level statistics to the row
        for grp in valid_groups:
            prefix = grp.replace(" ", "_")  # e.g. 'septic shock' -> 'septic_shock'
            row_dict[f"{prefix}_weighted_r"] = r_means.get(grp, np.nan)
            row_dict[f"{prefix}_se_r"]       = se_r_vals.get(grp, np.nan)
            row_dict[f"{prefix}_I2"]         = I2_vals.get(grp, np.nan)
            row_dict[f"{prefix}_z_stat"]     = z_stats.get(grp, np.nan)
            row_dict[f"{prefix}_p_value"]    = p_vals.get(grp, np.nan)
            row_dict[f"{prefix}_n_samples"] = sample_counts.get(grp, 0.0)

        # Differential correlation metrics between groups
        # Define helper to compute delta, z_diff and p_diff
        def compute_delta_zp(g1, g2):
            r1, r2 = r_means.get(g1, np.nan), r_means.get(g2, np.nan)
            se1, se2 = se_r_vals.get(g1, np.nan), se_r_vals.get(g2, np.nan)
            if all(np.isfinite([r1, r2, se1, se2])) and se1 + se2 > 0:
                delta = r1 - r2
                se_diff = math.sqrt(se1**2 + se2**2)
                z_diff = delta / se_diff if se_diff != 0 else np.nan
                p_diff = 2 * (1.0 - norm.cdf(abs(z_diff))) if np.isfinite(z_diff) else np.nan
                return delta, se_diff, z_diff, p_diff
            else:
                return (np.nan, np.nan, np.nan, np.nan)

        # Compute differences for shock-control, shock-sepsis, and sepsis-control
        delta_sc, se_sc, z_sc, p_sc = compute_delta_zp("septic shock", "control")
        delta_ss, se_ss_, z_ss_, p_ss_ = compute_delta_zp("septic shock", "sepsis")
        delta_sc2, se_sc2, z_sc2, p_sc2 = compute_delta_zp("sepsis", "control")

        row_dict.update({
            "delta_r_shock_control": delta_sc,
            "se_diff_shock_control": se_sc,
            "z_diff_shock_control": z_sc,
            "p_diff_shock_control": p_sc,
            "delta_r_shock_sepsis": delta_ss,
            "se_diff_shock_sepsis": se_ss_,
            "z_diff_shock_sepsis": z_ss_,
            "p_diff_shock_sepsis": p_ss_,
            "delta_r_sepsis_control": delta_sc2,
            "se_diff_sepsis_control": se_sc2,
            "z_diff_sepsis_control": z_sc2,
            "p_diff_sepsis_control": p_sc2,
        })

        # Combined statistics across all three groups
        # Combined p-value as geometric mean of group p-values
        p_list = [p_vals.get(g, np.nan) for g in valid_groups if np.isfinite(p_vals.get(g, np.nan))]
        if len(p_list) == 3:
            combined_p = math.sqrt(p_list[0] * p_list[1] * p_list[2])
        else:
            combined_p = np.nan
        # Combined effect size as geometric mean of absolute r values
        r_list = [abs(r_means.get(g, np.nan)) for g in valid_groups if np.isfinite(r_means.get(g, np.nan))]
        if len(r_list) == 3:
            combined_effect = math.sqrt(r_list[0] * r_list[1] * r_list[2])
        else:
            combined_effect = np.nan
        # Combined z statistic (root-sum-of-squares)
        z_list = [z_stats.get(g, np.nan) for g in valid_groups if np.isfinite(z_stats.get(g, np.nan))]
        if len(z_list) == 3:
            combined_z = math.sqrt(z_list[0]**2 + z_list[1]**2 + z_list[2]**2)
        else:
            combined_z = np.nan
        # Total sample size across all groups
        total_samples = sum(sample_counts.get(g, 0.0) for g in valid_groups)
        if np.isfinite(combined_z) and total_samples > 0:
            power_score = (combined_z ** 2) / ((combined_z ** 2) + total_samples)
        else:
            power_score = np.nan
        # Consistency metrics using mean I2 across groups (convert percent to proportion)
        I2_vals_list = [I2_vals.get(g, np.nan) for g in valid_groups if np.isfinite(I2_vals.get(g, np.nan))]
        if len(I2_vals_list) == 3:
            mean_I2 = np.mean([v / 100.0 for v in I2_vals_list])
            consistency_score = 1.0 / (1.0 + mean_I2)
        else:
            consistency_score = np.nan
        # Direction consistency: all r_means have same sign
        if all(np.isfinite([r_means.get(g, np.nan) for g in valid_groups])):
            signs = [math.copysign(1, r_means[g]) for g in valid_groups]
            direction_consistency = 1.0 if len(set(signs)) == 1 else 0.0
        else:
            direction_consistency = np.nan
        # Magnitude consistency: 1 - coefficient of variation of absolute r values
        if len(r_list) == 3 and np.mean(r_list) > 0:
            cv = np.std(r_list) / (np.mean(r_list) + 1e-10)
            magnitude_consistency = max(0.0, 1.0 - cv)
        else:
            magnitude_consistency = np.nan
        # Expression-based features using gene expression columns (if available)
        # We take first non-null values for geneA and geneB expression in sepsis and control
        try:
            geneA_sepsis = pair_df["geneA_ss_sepsis"].dropna().iloc[0] if not pair_df["geneA_ss_sepsis"].dropna().empty else np.nan
            geneA_ctrl   = pair_df["geneA_ss_ctrl"].dropna().iloc[0]   if not pair_df["geneA_ss_ctrl"].dropna().empty else np.nan
            geneB_sepsis = pair_df["geneB_ss_sepsis"].dropna().iloc[0] if not pair_df["geneB_ss_sepsis"].dropna().empty else np.nan
            geneB_ctrl   = pair_df["geneB_ss_ctrl"].dropna().iloc[0]   if not pair_df["geneB_ss_ctrl"].dropna().empty else np.nan
            # Compute log2 fold change; add small constant to avoid log of zero
            fc_a = np.log2((abs(geneA_sepsis) + 1e-10) / (abs(geneA_ctrl) + 1e-10)) if (np.isfinite(geneA_sepsis) and np.isfinite(geneA_ctrl)) else np.nan
            fc_b = np.log2((abs(geneB_sepsis) + 1e-10) / (abs(geneB_ctrl) + 1e-10)) if (np.isfinite(geneB_sepsis) and np.isfinite(geneB_ctrl)) else np.nan
            expression_change_mag = math.sqrt(fc_a**2 + fc_b**2) if all(np.isfinite([fc_a, fc_b])) else np.nan
            expression_asymmetry = max(abs(fc_a), abs(fc_b)) if all(np.isfinite([fc_a, fc_b])) else np.nan
        except Exception:
            fc_a = fc_b = expression_change_mag = expression_asymmetry = np.nan

        # Update row_dict with combined and expression features
        row_dict.update({
            "combined_p_value": combined_p,
            "combined_effect_size": combined_effect,
            "combined_z_score": combined_z,
            "power_score": power_score,
            "consistency_score": consistency_score,
            "direction_consistency": direction_consistency,
            "magnitude_consistency": magnitude_consistency,
            "geneA_log2fc": fc_a,
            "geneB_log2fc": fc_b,
            "expression_change_magnitude": expression_change_mag,
            "expression_asymmetry": expression_asymmetry,
            "total_samples": total_samples,
        })

        # Append row
        rows.append(row_dict)

    met = pd.DataFrame(rows)
    # FDR across all pairs
    if "p_ss" in met:
        met["q_ss"] = bh_fdr(met["p_ss"].values)
    if "p_soth" in met:
        met["q_soth"] = bh_fdr(met["p_soth"].values)
    
    logger.debug("Rows collected: %s", len(rows))
    logger.debug("Meta DataFrame shape: %s", met.shape)
    logger.debug("Meta DataFrame columns: %s", met.columns.tolist())

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

# ------------------ Unsupervised ranking ------------------

def compute_unsupervised_rank_scores(met: pd.DataFrame) -> pd.DataFrame:
    """
    Given the meta-analytic metrics DataFrame, compute unsupervised anomaly scores
    across all numeric features (excluding the existing rank_score) and derive
    a combined rank score.  The resulting DataFrame will include three new
    columns: unsupervised_score, metrics_score and new_rank_score, and will
    not drop any existing rows.
    """
    # Identify numeric columns and exclude the existing rank_score if present
    numeric_cols = met.select_dtypes(include=[np.number]).columns.tolist()
    if "rank_score" in numeric_cols:
        numeric_cols.remove("rank_score")

    # Build matrix of features for anomaly detection
    X = met[numeric_cols].copy()
    # Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    # Scale features to zero mean / unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    n_samples = X_scaled.shape[0]
    if n_samples < 2:
        logger.debug(
            "Insufficient samples (%s) for full unsupervised scoring; defaulting anomaly scores to zeros",
            n_samples,
        )
        iso_anomaly = np.zeros(n_samples, dtype=float)
        lof_scores = np.zeros(n_samples, dtype=float)
        distances = np.zeros(n_samples, dtype=float)
    else:
        # Compute anomaly scores using multiple unsupervised methods
        # Isolation Forest
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso_anomaly = -iso.fit(X_scaled).score_samples(X_scaled)
        # Local Outlier Factor
        n_neighbors = min(20, n_samples - 1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
        # Fit LOF on the scaled data; negative_outlier_factor_ is available after fit
        lof.fit(X_scaled)
        lof_scores = -lof.negative_outlier_factor_
        # KMeans distances to cluster centroids
        n_clusters = min(5, n_samples)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        centers = kmeans.cluster_centers_
        distances = np.linalg.norm(X_scaled - centers[labels], axis=1)

    # Helper for min-max normalisation
    def _minmax(arr: np.ndarray) -> np.ndarray:
        arr = np.array(arr, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        minv = np.nanmin(arr)
        maxv = np.nanmax(arr)
        return (arr - minv) / (maxv - minv + 1e-12)

    # Normalised scores from each model
    if_norm = _minmax(iso_anomaly)
    lof_norm = _minmax(lof_scores)
    dist_norm = _minmax(distances)
    # Aggregate unsupervised score
    unsupervised_score = (if_norm + lof_norm + dist_norm) / 3.0

    # Compute domain-specific metrics score
    met_copy = met.copy()
    # -log10 of q-values (larger is better)
    met_copy["neglog_q_ss"] = -np.log10(met_copy["q_ss"].clip(lower=1e-300))
    met_copy["neglog_q_soth"] = -np.log10(met_copy["q_soth"].clip(lower=1e-300))
    # Ensure all required columns exist and fill missing with zero
    for col in [
        "abs_dz_ss",
        "abs_dz_soth",
        "kappa_ss",
        "kappa_soth",
        "combined_effect_size",
        "power_score",
        "consistency_score",
        "direction_consistency",
        "magnitude_consistency",
    ]:
        if col not in met_copy.columns:
            met_copy[col] = np.nan
        met_copy[col] = met_copy[col].fillna(0)
    feature_list = [
        "neglog_q_ss",
        "abs_dz_ss",
        "kappa_ss",
        "neglog_q_soth",
        "abs_dz_soth",
        "kappa_soth",
        "combined_effect_size",
        "power_score",
        "consistency_score",
        "direction_consistency",
        "magnitude_consistency",
    ]
    values = met_copy[feature_list].values
    feat_norm = np.zeros_like(values, dtype=float)
    for i, col in enumerate(feature_list):
        feat_norm[:, i] = _minmax(values[:, i])
    metrics_score = feat_norm.mean(axis=1)

    # Final new rank score: average of unsupervised and domain metrics scores
    new_rank_score = (unsupervised_score + metrics_score) / 2.0

    # Attach scores back to DataFrame
    met_copy["unsupervised_score"] = unsupervised_score
    met_copy["metrics_score"] = metrics_score
    met_copy["new_rank_score"] = new_rank_score
    return met_copy

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
    logger.info("Computing meta-analytic metrics...")
    met = compute_metrics(df, labels)
    logger.info(f"Computed metrics for {len(met)} gene pairs")

    # 3) (Optional) join gene names if you want them in the CSV
    name_cols = df.groupby("pair_id")[["GeneAName","GeneBName","GeneAKey","GeneBKey"]].first().reset_index()
    met = met.merge(name_cols, on="pair_id", how="left")
    logger.info("Merged gene names")

    # 4) Prepare numeric columns for unsupervised learning
    numeric_cols = met.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        nan_counts = met[numeric_cols].isna().sum().sum()
        if nan_counts:
            logger.debug("Filling %s NaN values across numeric columns before unsupervised scoring", nan_counts)
        met[numeric_cols] = met[numeric_cols].fillna(0.0)
    else:
        logger.debug("No numeric columns detected prior to unsupervised scoring")

    # 5) Compute unsupervised ranking and append scores
    logger.info("Computing unsupervised rank scores...")
    met = compute_unsupervised_rank_scores(met)
    logger.info("Unsupervised scores computed")

    # Sort by new rank score descending for final output
    met = met.sort_values("new_rank_score", ascending=False).reset_index(drop=True)
    logger.info("Sorted gene pairs by new rank score")

    # 5) Save
    met.to_csv(args.out, index=False)
    logger.info(f"Wrote {args.out}  rows={len(met)}")

if __name__ == "__main__":
    main()
