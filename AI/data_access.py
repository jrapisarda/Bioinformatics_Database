import pandas as pd
from sqlalchemy import text

# 1) Studywise correlations by illness (your existing view/table)
def load_studywise_corr(engine, labels):
    # expect columns: study_key, gene_a_key, gene_b_key, illness_key, r (Spearman), n
    # If you already materialize these in a view/table, point here:
    sql = text("""
        SELECT study_key, GeneAKey AS gene_a_key, GeneBKey AS gene_b_key,
               illness_label as illness_key, rho_spearman AS r
        FROM dbo.vw_gene_DE_fact_corr_data  -- adjust if needed
    """)
    df = pd.read_sql(sql, engine)
    # Make a stable pair_id = min_max
    pa, pb = df["gene_a_key"].astype(int), df["gene_b_key"].astype(int)
    df["pair_id"] = (pa.where(pa<=pb, pb).astype(str) + "_" + pb.where(pa<=pb, pa).astype(str))
    # Label names
    inv = {v:k for k,v in labels.items()}
    df["group"] = df["illness_key"].map(inv).str.replace("_", " ")
    return df

# 2) Per-sample expressions for LOSO & permutations
def load_pair_samples(engine, gene_a, gene_b):
    # Requires: fact_expression(sample_key, gene_key, expression_value), dim_sample(sample_key, study_key, illness_key)
    sql = text("""
        WITH ex AS (
            SELECT s.study_key, s.sample_key, s.illness_key,
                   MAX(CASE WHEN f.gene_key = :ga THEN f.expression_value END) AS expr_a,
                   MAX(CASE WHEN f.gene_key = :gb THEN f.expression_value END) AS expr_b
            FROM dbo.fact_expression f
            JOIN dbo.dim_sample s ON s.sample_key = f.sample_key
            WHERE f.gene_key IN (:ga,:gb)
            GROUP BY s.study_key, s.sample_key, s.illness_key
        )
        SELECT * FROM ex
    """)
    return pd.read_sql(sql, engine, params={"ga": int(gene_a), "gb": int(gene_b)})
