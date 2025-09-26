import os, json
import pandas as pd
from openai import OpenAI

# --- Config ---
#
# The script has been extended to operate on the entire metrics CSV and
# score each row using all available numeric columns.  Results from the
# model (verdict, rationale, flags etc.) are stored in a new column on
# the dataframe and the augmented data is written back out to disk.  You
# can customize the input and output CSV paths below, and the model name
# if you prefer a different OpenAI engine.

CSV_PATH = "pair_metrics_agent.csv"  # input CSV file
# Name of the output CSV with the new result column. The script will
# write the original data plus a new column to this file.
OUT_CSV_PATH = "pair_metrics_agent_scored.csv"
# Name of the OpenAI model to query.  Adjust as needed.  See OpenAI docs
# for available models.
MODEL = "gpt-5-mini"

# Optional: include extra identifier columns in the payload alongside
# numeric columns.  Gene names and IDs can aid the model in reasoning
# but are not required.  If you wish to omit them, set EXTRA_ID_COLS = []
EXTRA_ID_COLS = ["pair_id", "GeneAName", "GeneBName", "GeneAKey", "GeneBKey"]

# The PAIR_ID constant from the previous version has been removed; the script
# no longer targets a single row but processes the entire input file.

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

# --- Load and prepare the dataframe ---
#
# Read the input CSV.  We expect a 'pair_id' column for context, but
# otherwise accept any schema.  All numeric columns will be fed to the
# model for evaluation.  If you need to limit the rows processed you
# can slice `df` here.
df = pd.read_csv(CSV_PATH)
if df.empty:
    raise ValueError(f"Input CSV {CSV_PATH} appears to be empty.")
if "pair_id" not in df.columns:
    raise ValueError(f"{CSV_PATH} must contain a 'pair_id' column.")

# Identify all numeric columns in the dataframe.  We use Pandas' type
# inference to include ints, floats and booleans.  You can adjust the
# `include` argument if additional dtypes are needed.
numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()

# Build the list of columns to include in the payload sent to the model.
# Start with all numeric columns and append extra identifiers if present.
cols_to_use = numeric_cols.copy()
for col in EXTRA_ID_COLS:
    if col in df.columns and col not in cols_to_use:
        cols_to_use.append(col)

# Precompute the messages that are constant across all rows.  The rubric
# does not change per row.
rubric = """You are a bioinformatics PhD agent. Classify biomarker feasibility for a gene pair using these rules:
- GOOD: strong effect and significance with consistency and acceptable heterogeneity.
  Typical thresholds (tuneable): q_ss ≤ 0.05, |dz_ss_mean| ≥ 0.40, kappa_ss ≥ 0.70, I²_ss < 60%.
- MODERATE: promising signal but one or two weaknesses (e.g., borderline q or effect, limited studies, or moderate heterogeneity).
- NEGATIVE: fails significance or effect thresholds, low consistency, very high heterogeneity (I² ≥ 75%), or too few studies.
Use Shock vs Sepsis (SS) as primary evidence; Shock vs Others (SOTH) is supportive. Consider CIs and heterogeneity (Q/I²) in your risk assessment.
Return JSON with: verdict (Good|Moderate|Negative), rationale (1–3 concise bullets), red_flags (array), supporting_metrics (array of 'name: value' strings), recommendations (array).
"""

# Instantiate the OpenAI client.  This will pick up your API key from
# environment variable OPENAI_API_KEY.
client = OpenAI()

# Container to collect the full JSON results.  Storing the full JSON
# allows downstream users to inspect rationale and other metadata.  If
# you only care about the verdict you can instead extract result["verdict"].
full_results = []

# Iterate through each row and call the model.  Each payload is built
# separately to ensure per-row values are passed.  This loop may take
# time proportional to the number of rows and network latency.
for idx, row in df.iterrows():
    # Build the JSON-like dictionary containing only the selected columns.
    row_payload = jsonify({k: (row[k] if k in row.index else None) for k in cols_to_use})

    # Construct the messages for this request.  We include the rubric as
    # the system message and the JSON payload as the user message.  The
    # payload is inserted into a fenced code block to preserve structure.
    messages = [
        {"role": "system", "content": rubric},
        {
            "role": "user",
            "content": (
                "Evaluate this gene-pair metrics row and classify biomarker feasibility.\n"
                "Input (JSON):\n```json\n" + json.dumps(row_payload, indent=2) + "\n```"
            ),
        },
    ]

    # Call the OpenAI chat completion endpoint asking for structured JSON.
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        max_completion_tokens=4000,
    )

    # Parse the JSON response.  We expect result to be a dictionary with
    # keys: verdict, rationale, red_flags, supporting_metrics, recommendations.
    result = json.loads(resp.choices[0].message.content)
    full_results.append(result)

    # Store the result JSON string into a new column on the dataframe.  We
    # serialise the dictionary to JSON so it can be saved to CSV without
    # losing structure.  You could instead store result["verdict"] if you
    # only care about the high-level classification.
    df.loc[idx, "agent_result_json"] = json.dumps(result)

# After processing all rows, write the augmented dataframe back out.
df.to_csv(OUT_CSV_PATH, index=False)

# Optionally print a summary to stdout.  This can help during
# interactive runs to see the classifications without opening the CSV.
print(f"Processed {len(df)} rows.  Results have been written to {OUT_CSV_PATH}.")
