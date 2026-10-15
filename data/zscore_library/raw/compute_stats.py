import pandas as pd
import os

RAW_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(RAW_DIR, "results")
OUTPUT_DIR = os.path.join(RAW_DIR, "..", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_target_stats(scores_df):
    """Return DataFrame with target_id, mean_score, std_score per column."""
    return pd.DataFrame({
        "target_id": scores_df.columns,
        "mean_score": scores_df.mean(axis=0, skipna=True).values,
        "std_score": scores_df.std(axis=0, skipna=True).values,
    })


def chain_id_to_col_key(pdb_id, pdb_chain_id):
    """Convert safety_targets.csv chain notation to scores CSV column key."""
    chain_str = str(pdb_chain_id).strip()
    if chain_str.startswith("[") and chain_str.endswith("]"):
        chain_str = chain_str[1:-1].replace(";", "_")
    return f"{pdb_id}_{chain_str}"


# ── Case studies ──────────────────────────────────────────────────────────────
cs_scores = pd.read_csv(
    os.path.join(RESULTS_DIR, "case_studies_scores.csv"), index_col="lig_idx"
)
cs_stats = compute_target_stats(cs_scores)
cs_stats.to_csv(os.path.join(OUTPUT_DIR, "case_studies_stats.csv"), index=False)
print(f"case_studies_stats.csv  →  {len(cs_stats)} targets")

# ── Standard docking benchmark  (target_id = case_name) ──────────────────────
sdb_scores = pd.read_csv(
    os.path.join(RESULTS_DIR, "standard_docking_benchmark_scores.csv"),
    index_col="lig_idx",
)

sdb_meta = pd.read_csv(
    os.path.join(RAW_DIR, "../../standard_docking_benchmark/standard_docking_benchmark.csv")
)
sdb_meta["col_key"] = sdb_meta.apply(
    lambda r: chain_id_to_col_key(r["pdb_id"], r["pdb_chain_id"]), axis=1
)
sdb_col_to_case_name = dict(zip(sdb_meta["col_key"], sdb_meta["case_name"]))

sdb_stats = compute_target_stats(sdb_scores)
sdb_stats["target_id"] = sdb_stats["target_id"].map(sdb_col_to_case_name)

unmapped = sdb_stats["target_id"].isna().sum()
if unmapped:
    print(f"WARNING: {unmapped} standard docking benchmark column(s) could not be mapped to a case_name")

sdb_stats.to_csv(
    os.path.join(OUTPUT_DIR, "standard_docking_benchmark_stats.csv"), index=False
)
print(f"standard_docking_benchmark_stats.csv  →  {len(sdb_stats)} targets")

# ── Safety targets  (target_id = case_name) ───────────────────────────────────
st_scores = pd.read_csv(
    os.path.join(RESULTS_DIR, "safety_targets_scores.csv"), index_col="lig_idx"
)

st_meta = pd.read_csv(
    os.path.join(RAW_DIR, "../../safety_targets/safety_targets.csv")
)
st_meta["col_key"] = st_meta.apply(
    lambda r: chain_id_to_col_key(r["pdb_id"], r["pdb_chain_id"]), axis=1
)
col_to_case_name = dict(zip(st_meta["col_key"], st_meta["case_name"]))

st_stats = compute_target_stats(st_scores)
st_stats["target_id"] = st_stats["target_id"].map(col_to_case_name)

unmapped = st_stats["target_id"].isna().sum()
if unmapped:
    print(f"WARNING: {unmapped} safety target column(s) could not be mapped to a case_name")

st_stats.to_csv(os.path.join(OUTPUT_DIR, "safety_targets_stats.csv"), index=False)
print(f"safety_targets_stats.csv  →  {len(st_stats)} targets")

# ── CrossDocked test set  (target_id = pdb_id_chain, matches _dock_molecules keys) ──
xd_scores = pd.read_csv(
    os.path.join(RESULTS_DIR, "crossdocked_test_set_scores.csv"), index_col="lig_idx"
)
xd_stats = compute_target_stats(xd_scores)
# Column names from aggregate_results.py are already pdb_id_chain (e.g. "14gs_A"),
# matching the on_key/off_key convention used by _dock_molecules in dock.py.
xd_stats.to_csv(os.path.join(OUTPUT_DIR, "crossdocked_test_set_stats.csv"), index=False)
print(f"crossdocked_test_set_stats.csv  →  {len(xd_stats)} targets")
