import csv
import itertools

matrix_path = "../processed/probis_zscore_matrix.csv"
output_path = "../crossdocked_test_set_pairs.csv"

# Parse matrix
with open(matrix_path) as f:
    reader = csv.reader(f)
    rows = list(reader)

# First row: header (first cell empty, rest are column labels)
col_labels = rows[0][1:]  # skip the empty corner cell
n = len(col_labels)

# Build full symmetric matrix (using max of upper/lower if both present)
# rows[1:] are data rows: first cell = row label, rest = values
matrix = {}
for row in rows[1:]:
    row_label = row[0]
    values = row[1:]
    matrix[row_label] = {}
    for j, col_label in enumerate(col_labels):
        try:
            matrix[row_label][col_label] = float(values[j])
        except (IndexError, ValueError):
            matrix[row_label][col_label] = 0.0

def difficulty(z):
    if z > 3:
        return "hard"
    elif z >= 2:
        return "medium"
    else:
        return "easy"

def parse_pdb_chain(label):
    # e.g. "2z3hA" -> pdb_id="2z3h", chain_id="A"
    # labels are 5 chars: 4-char PDB ID + 1-char chain
    if len(label) >= 5:
        return label[:4], label[4:]
    return label, ""

pairs = []
case_id = 0
labels = col_labels  # all unique protein labels

for i, j in itertools.combinations(range(n), 2):
    label_i = labels[i]
    label_j = labels[j]
    # Use the value from the matrix; prefer non-zero from either direction
    z = matrix.get(label_i, {}).get(label_j, 0.0)
    if z == 0.0:
        z = matrix.get(label_j, {}).get(label_i, 0.0)

    pdb1_id, pdb1_chain = parse_pdb_chain(label_i)
    pdb2_id, pdb2_chain = parse_pdb_chain(label_j)

    diff = difficulty(z)

    # Permutation 1: on_target = pdb1
    pairs.append({
        "case_id": case_id,
        "on_target": "pdb1",
        "pdb1_id": pdb1_id,
        "pdb1_chain_id": pdb1_chain,
        "pdb2_id": pdb2_id,
        "pdb2_chain_id": pdb2_chain,
        "difficulty": diff,
    })
    case_id += 1

    # Permutation 2: on_target = pdb2
    pairs.append({
        "case_id": case_id,
        "on_target": "pdb2",
        "pdb1_id": pdb1_id,
        "pdb1_chain_id": pdb1_chain,
        "pdb2_id": pdb2_id,
        "pdb2_chain_id": pdb2_chain,
        "difficulty": diff,
    })
    case_id += 1

difficulty_order = {"hard": 0, "medium": 1, "easy": 2}
pairs.sort(key=lambda p: difficulty_order[p["difficulty"]])

# Reassign case_id after sorting
for idx, p in enumerate(pairs):
    p["case_id"] = idx

with open(output_path, "w", newline="") as f:
    fieldnames = ["case_id", "on_target", "pdb1_id", "pdb1_chain_id", "pdb2_id", "pdb2_chain_id", "difficulty"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(pairs)

print(f"Written {len(pairs)} pairs to {output_path}")
# Quick stats
hard = sum(1 for p in pairs if p["difficulty"] == "hard")
medium = sum(1 for p in pairs if p["difficulty"] == "medium")
easy = sum(1 for p in pairs if p["difficulty"] == "easy")
print(f"hard: {hard}, medium: {medium}, easy: {easy}")
