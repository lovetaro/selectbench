import csv
import math
import os
from rdkit import Chem


def load_case_studies(csv_path, row_key="case_name"):
    cases = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases[row[row_key]] = row
    return cases


def load_reference_stats(csv_path):
    """Load per-target reference statistics for Z-score normalization.

    CSV must have columns: target_id, mean_score, std_score
    target_id should match '{pdb_id}_{chain_id}' for case study targets,
    or 'case_name' for safety targets.

    Returns dict: {target_id: (mean_score, std_score)}
    """
    stats = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats[row["target_id"]] = (float(row["mean_score"]), float(row["std_score"]))
    return stats


def _stats_from_scores(scores):
    """Return (mean, std) or (None, None) if fewer than 2 valid scores."""
    if len(scores) < 2:
        return None, None
    n = len(scores)
    mean = sum(scores) / n
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / (n - 1))
    return mean, std


def _load_smiles_list(sdf_file, smiles_file):
    if sdf_file:
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=True)
        return [Chem.MolToSmiles(mol) for mol in suppl if mol is not None]
    with open(smiles_file) as f:
        return [line.strip() for line in f if line.strip()]


def _append_stats_row(output_path, target_id, mean, std):
    file_exists = os.path.exists(output_path)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target_id", "mean_score", "std_score"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"target_id": target_id, "mean_score": mean, "std_score": std})
