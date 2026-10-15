import os
import pandas as pd
import re

# =========================
CSV = "./crossdocked_test_set.csv"
RESULTS_DIR = "./probis_results"
OUT_CSV = "../processed/probis_zscore_matrix.csv"
# =========================

print("Reading CSV...")

df = pd.read_csv(CSV)

probis_ids = df["case_name"].tolist()
probis_ids = [id.replace("_", "") for id in probis_ids]

print(f"Proteins: {len(probis_ids)}")

# Create matrix
matrix = pd.DataFrame(
    0.0,
    index=probis_ids,
    columns=probis_ids
)

print("Scanning .0.rota.pdb files...")

pattern = re.compile(r"REMARK PROBIS Z_SCORE\s+([-0-9.]+)")

for file in os.listdir(RESULTS_DIR):

    # Only consider .0.rota.pdb
    if not file.endswith(".0.rota.pdb"):
        continue

    full_path = os.path.join(RESULTS_DIR, file)

    # filename format: 1a2gA_1djyA.0.rota.pdb
    base = file.replace(".0.rota.pdb", "")

    try:
        id1, id2 = base.split("_")
    except:
        continue

    if id1 not in probis_ids or id2 not in probis_ids:
        continue

    with open(full_path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                z = float(match.group(1))

                matrix.loc[id1, id2] = z
                # matrix.loc[id2, id1] = z # Symmetric, but we only fill upper triangle to avoid duplicates
                break

print("Saving matrix...")

matrix.to_csv(OUT_CSV)

print("Done.")
print(f"Saved to {OUT_CSV}")