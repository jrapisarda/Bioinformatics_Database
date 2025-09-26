import argparse
import ast
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------- helpers ----------

def _literal_coerce(s):
    """
    Try to parse a string like "1.23", "2.0e-6", "[1,2]" into a Python number/list.
    Falls back to regex-based number extraction. Returns original if nothing found.
    """
    if not isinstance(s, str):
        return s
    s_strip = s.strip()
    # Try safe literal eval first (handles lists and numbers)
    try:
        v = ast.literal_eval(s_strip)
        return v
    except Exception:
        pass
    # If looks like a plain number, parse it
    num_match = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?", s_strip)
    if not num_match:
        return s
    if len(num_match) == 1:
        try:
            return float(num_match[0])
        except Exception:
            return s
    # multiple numbers -> list of floats
    out = []
    for tok in num_match:
        try:
            out.append(float(tok))
        except Exception:
            pass
    return out if out else s

def parse_supporting_metrics(metrics_list):
    """
    Parse the model's supporting_metrics (list of "name: value" strings)
    into a dict of {name: parsed_value}.
    """
    parsed = {}
    for item in metrics_list:
        if not isinstance(item, str):
            continue
        if ":" in item:
            k, v = item.split(":", 1)
            k = k.strip()
            v = _literal_coerce(v.strip())
            parsed[k] = v
        else:
            # no colon; store raw
            parsed[item.strip()] = None
    return parsed

def get_float(d, key, default=None):
    v = d.get(key, default)
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    if isinstance(v, list) and len(v) > 0:
        # if list provided where scalar expected, take first numeric
        for vi in v:
            if isinstance(vi, (int, float)) and math.isfinite(vi):
                return float(vi)
        return default
    # try to coerce strings
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?", v)
            if nums:
                try:
                    return float(nums[0])
                except Exception:
                    return default
    return default

def get_ci(d, key, low_key=None, high_key=None):
    """
    Return (low, high) 95% CI from either:
      - a single key like "dz_ss_ci": [low, high]
      - or separate low/high keys (if provided)
    """
    if key in d and isinstance(d[key], list) and len(d[key]) >= 2:
        lo, hi = d[key][0], d[key][1]
        return float(lo), float(hi)
    if low_key and high_key and (low_key in d or high_key in d):
        lo = get_float(d, low_key)
        hi = get_float(d, high_key)
        return lo, hi
    return (None, None)

def neglog10(x):
    """safe -log10 for p/q"""
    if x is None or not math.isfinite(x) or x <= 0:
        return 0.0
    return -math.log10(x)

# ---------- plotting pages ----------

def page_summary(pdf, pair_id, verdict, rationale, red_flags, recs):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    title = f"Biomarker Pair Report: {pair_id or '(unknown pair)'}"
    verdict_txt = f"Model Verdict: {verdict or 'N/A'}"
    ax.text(0.06, 0.95, title, fontsize=18, weight="bold")
    ax.text(0.06, 0.90, verdict_txt, fontsize=14, weight="bold")

    def bullets(y, header, items):
        ax.text(0.06, y, header, fontsize=12, weight="bold")
        y -= 0.02
        if not items:
            ax.text(0.08, y, "– None", fontsize=10)
            return y - 0.02
        for it in items:
            ax.text(0.08, y, f"• {it}", fontsize=10, wrap=True)
            y -= 0.02
        return y - 0.01

    y = 0.85
    y = bullets(y, "Rationale", rationale or [])
    y = bullets(y, "Red Flags", red_flags or [])
    y = bullets(y, "Recommendations", recs or [])

    pdf.savefig(fig); plt.close(fig)

def page_forest(pdf, m):
    """
    Two-row forest plot for SS and SOTH using dz mean and 95% CI.
    """
    rows = [
        ("Shock vs Sepsis",  "dz_ss_mean",  "dz_ss_ci",  "p_ss",  "q_ss"),
        ("Shock vs Others",  "dz_soth_mean","dz_soth_ci","p_soth","q_soth"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    y_positions = [1, 0]
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)

    labels = []
    for (label, mean_key, ci_key, p_key, q_key), y in zip(rows, y_positions):
        mu = get_float(m, mean_key)
        lo, hi = get_ci(m, ci_key)
        if mu is None or lo is None or hi is None:
            continue
        ax.errorbar(mu, y, xerr=[[mu - lo], [hi - mu]], fmt="o", capsize=5)
        labels.append(label)
        # Annotate to the right
        p = get_float(m, p_key); q = get_float(m, q_key)
        txt = f"{label}: Δz={mu:.3f}  95%CI[{lo:.3f}, {hi:.3f}]  p={p:.2g}  q={q:.2g}"
        ax.text(hi + 0.05, y, txt, va="center", fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels if labels else ["Shock vs Sepsis","Shock vs Others"])
    ax.set_xlabel("Pooled Fisher z difference (Δz)")
    ax.set_title("Effect size (random-effects) with 95% CI")
    pdf.savefig(fig); plt.close(fig)

def page_pvals(pdf, m):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    pvals = [get_float(m, "p_ss"), get_float(m, "p_soth")]
    qvals = [get_float(m, "q_ss"), get_float(m, "q_soth")]
    cats  = ["SS", "SOTH"]

    x = [0, 1]
    width = 0.35
    ax.bar([i - width/2 for i in x], [neglog10(p) for p in pvals], width=width, label="-log10(p)")
    ax.bar([i + width/2 for i in x], [neglog10(q) for q in qvals], width=width, label="-log10(q)")

    # q=0.05 threshold line
    ax.axhline(-math.log10(0.05), linestyle="--", linewidth=1, alpha=0.6)
    ax.text(1.02, -math.log10(0.05), " q=0.05", va="center", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("Enrichment (-log10)")
    ax.set_title("Significance summary")
    ax.legend()
    pdf.savefig(fig); plt.close(fig)

def page_heterogeneity(pdf, m):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    vals = [get_float(m, "dz_ss_I2"), get_float(m, "dz_soth_I2")]
    cats = ["SS", "SOTH"]
    ax.bar(range(len(cats)), [v if v is not None else 0 for v in vals])
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats)
    ax.set_ylim(0, 100)
    # annotate thresholds
    for thr, label in [(60, "moderate/high"), (75, "high")]:
        ax.axhline(thr, linestyle="--", linewidth=1, alpha=0.5)
        ax.text(len(cats)-0.4, thr+1, f"I² {label}", fontsize=9)
    ax.set_ylabel("I² (%)")
    ax.set_title("Between-study heterogeneity")
    pdf.savefig(fig); plt.close(fig)

def page_consistency_and_counts(pdf, m):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    kappas = [get_float(m, "kappa_ss"), get_float(m, "kappa_soth")]
    nstuds = [get_float(m, "n_studies_ss"), get_float(m, "n_studies_soth")]
    x = [0, 1]
    width = 0.35
    ax.bar([i - width/2 for i in x], [k if k is not None else 0 for k in kappas], width=width, label="kappa (consistency)")
    ax.bar([i + width/2 for i in x], [n if n is not None else 0 for n in nstuds], width=width, label="n_studies")

    # guideline lines
    ax.axhline(0.7, linestyle="--", linewidth=1, alpha=0.5); ax.text(1.02, 0.7, "κ=0.7", va="center", fontsize=9)
    ax.axhline(5, linestyle="--", linewidth=1, alpha=0.5); ax.text(1.02, 5, "n=5", va="center", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(["SS","SOTH"])
    ax.set_title("Consistency & replication")
    ax.legend()
    pdf.savefig(fig); plt.close(fig)

def page_metrics_raw(pdf, supporting_metrics, verdict):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.06, 0.95, "Raw supporting metrics (from model)", fontsize=14, weight="bold")
    if verdict:
        ax.text(0.06, 0.92, f"Verdict: {verdict}", fontsize=12)
    # Render monospaced block
    y = 0.88
    for line in supporting_metrics:
        ax.text(0.06, y, line, fontsize=9, family="monospace")
        y -= 0.018
        if y < 0.05:
            break
    pdf.savefig(fig); plt.close(fig)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=False, help="Path to JSON file with model response. If omitted, reads stdin.")
    ap.add_argument("--out", required=True, help="Output PDF path")
    ap.add_argument("--pair-id", default=None, help="Optional pair_id to show in title")
    args = ap.parse_args()

    # Load JSON (stdin or file)
    if args.inp:
        data = json.loads(Path(args.inp).read_text())
    else:
        import sys
        data = json.load(sys.stdin)

    verdict = data.get("verdict")
    rationale = data.get("rationale", [])
    red_flags = data.get("red_flags", [])
    recs = data.get("recommendations", [])
    supporting_metrics = data.get("supporting_metrics", [])

    # Parse metrics list -> dict
    met_dict = parse_supporting_metrics(supporting_metrics)

    # Build PDF
    with PdfPages(args.out) as pdf:
        page_summary(pdf, args.pair_id, verdict, rationale, red_flags, recs)
        page_forest(pdf, met_dict)
        page_pvals(pdf, met_dict)
        page_heterogeneity(pdf, met_dict)
        page_consistency_and_counts(pdf, met_dict)
        page_metrics_raw(pdf, supporting_metrics, verdict)

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
