import os, json
import pandas as pd
from openai import OpenAI

# --- Config ---
CSV_PATH = "pair_metrics_agent.csv"
PAIR_ID  = "11_110"           # <-- change or parametrize as needed
MODEL    = "gpt-5-mini"       # reasoning mini model; fast + cheap for this task

def jsonify(obj):
    """Recursively cast NumPy/Pandas scalars to native Python types."""
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonify(i) for i in obj]
    if pd.isna(obj):
        return None
    if hasattr(obj, "item"):
        return obj.item()
    return obj

# --- Load the row ---
df = pd.read_csv(CSV_PATH)
if "pair_id" not in df.columns:
    raise ValueError("pair_metrics_agent.csv must contain a 'pair_id' column.")
row_df = df.loc[df["pair_id"] == PAIR_ID]
if row_df.empty:
    raise ValueError(f"Pair {PAIR_ID} not found in {CSV_PATH}.")
row = row_df.iloc[0]

# (Optional) restrict to the most informative columns to cut tokens
preferred_cols = [
    # identifiers
    "pair_id","GeneAName","GeneBName","GeneAKey","GeneBKey",
    # primary endpoint: Shock vs Sepsis (SS)
    "dz_ss_mean","dz_ss_ci_low","dz_ss_ci_high","dz_ss_I2","dz_ss_Q","p_ss","q_ss","kappa_ss","n_studies_ss",
    # secondary endpoint: Shock vs Others (SOTH)
    "dz_soth_mean","dz_soth_ci_low","dz_soth_ci_high","dz_soth_I2","dz_soth_Q","p_soth","q_soth","kappa_soth","n_studies_soth",
    # any extras your CSV includes (safe to ignore if missing)
    "auroc","brier","ece"
]
payload = jsonify({k: (row[k] if k in row.index else None)
                   for k in preferred_cols if k in row.index})

# --- Build messages ---
rubric = """You are a bioinformatics PhD agent. Classify biomarker feasibility for a gene pair using these rules:
- GOOD: strong effect and significance with consistency and acceptable heterogeneity.
  Typical thresholds (tuneable): q_ss ≤ 0.05, |dz_ss_mean| ≥ 0.40, kappa_ss ≥ 0.70, n_studies_ss ≥ 5, I²_ss < 60%.
- MODERATE: promising signal but one or two weaknesses (e.g., borderline q or effect, limited studies, or moderate heterogeneity).
- NEGATIVE: fails significance or effect thresholds, low consistency, very high heterogeneity (I² ≥ 75%), or too few studies.
Use Shock vs Sepsis (SS) as primary evidence; Shock vs Others (SOTH) is supportive. Consider CIs and heterogeneity (Q/I²) in your risk assessment.
Return JSON with: verdict (Good|Moderate|Negative), rationale (1–3 concise bullets), red_flags (array), supporting_metrics (array of 'name: value' strings), recommendations (array).
"""

messages = [
    {"role": "system", "content": rubric},
    {
        "role": "user",
        "content": (
            "Evaluate this gene-pair metrics row and classify biomarker feasibility.\n"
            "Input (JSON):\n```json\n" + json.dumps(payload, indent=2) + "\n```"
        ),
    },
]

# --- Call OpenAI Chat Completions (structured JSON out) ---
client = OpenAI()  # reads OPENAI_API_KEY
resp = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    response_format={"type": "json_object"},  # ask for strict JSON back
    max_completion_tokens=4000,
)

result = json.loads(resp.choices[0].message.content)
print(json.dumps(result, indent=2))
