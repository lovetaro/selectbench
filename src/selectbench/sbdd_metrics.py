"""Molecular quality metrics for generated molecules.

Implements MoleculeProperties from DiffSBDD
(https://github.com/arneschneuing/DiffSBDD/blob/main/analysis/metrics.py):
QED, SA score, LogP, Lipinski rule count, and intra-set diversity.

These metrics are **reference-free** — a pure function of the generated SMILES,
independent of docking or on-/off-target pairing — so they are distinct from the
selectivity metrics in ``selectbench.metrics``. They are surfaced in two opt-in
ways (RDKit is only imported on these paths, never on the default selectivity
pipeline):

* ``selectbench quality <file>`` — a docking-free CLI command, and
* ``scripts/summarize/cross_method.py --quality`` — extra columns in the
  cross-method summary, computed from the ``smiles`` column already stored in
  every raw results CSV.

Main entry points
-----------------
load_smiles_from_file(path)             -> list[(mol_idx, smiles)]
compute_per_mol_metrics(smiles_list)    -> list[dict]  (one dict per molecule)
compute_aggregate_metrics(per_mol_rows) -> dict         (one aggregated row)
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import pstdev
from typing import Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED
from rdkit.Contrib.SA_Score import sascorer


# ── SMILES file loading ───────────────────────────────────────────────────────

def load_smiles_from_file(path: str | Path) -> list[tuple[int, str]]:
    """Load SMILES from a .sdf, .smi/.txt (one per line), or .csv file.

    Returns ``(mol_idx, smiles)`` pairs where ``mol_idx`` is the record's
    **original position in the file**, so it lines up 1-to-1 with a Vina eval
    on the same file. Unparseable SDF records are dropped, which leaves gaps in
    ``mol_idx`` — exactly as ``selectbench evaluate-*`` does.

    * ``.sdf``      — parsed with RDKit; each molecule's canonical SMILES.
                      Mirrors ``_dock_sdf``: enumerate first, then drop the
                      records RDKit could not parse.
    * ``.smi``/``.txt`` (or any other extension) — one SMILES per line, taken
                      verbatim; only blank lines are skipped. Mirrors
                      ``_dock_smiles_file``. Note this means a ``#`` comment
                      line is treated as an (invalid) SMILES rather than
                      skipped, because that is what the eval path does and
                      alignment is the stronger guarantee.
    * ``.csv``      — tries column names 'smiles', 'SMILES', 'Smiles' in order;
                      ``mol_idx`` is the 0-based data-row position.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        suppl = Chem.SDMolSupplier(str(path))
        return [(i, Chem.MolToSmiles(mol))
                for i, mol in enumerate(suppl) if mol is not None]
    if suffix == ".csv":
        out: list[tuple[int, str]] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                for col in ("smiles", "SMILES", "Smiles"):
                    # DictReader fills missing trailing fields with None, so the
                    # key can exist with a None value and a ""-default get()
                    # would still hand back None.
                    s = (row.get(col) or "").strip()
                    if s:
                        out.append((i, s))
                        break
        return out
    with open(path) as f:
        return [(i, line.strip()) for i, line in enumerate(f) if line.strip()]


# ── Per-molecule property helpers ─────────────────────────────────────────────

def _normalized_sa(mol) -> float:
    """SA score mapped to [0, 1] where 1 = easiest to synthesize.

    Mirrors the normalization used in pocket2mol:
        sa_norm = (10 - raw_sa) / 9
    """
    raw = sascorer.calculateScore(mol)
    return round((10 - raw) / 9, 4)


def _molecular_weight(mol) -> float:
    return round(Descriptors.MolWt(mol), 2)


def _lipinski_count(mol) -> int:
    """Count how many of the 5 extended Lipinski rules are satisfied (0–5).

    Rules: MW < 500, HBD <= 5, HBA <= 10, -2 <= logP <= 5, RotBonds <= 10
    """
    logp = Crippen.MolLogP(mol)
    rules = [
        Descriptors.ExactMolWt(mol) < 500,
        Lipinski.NumHDonors(mol) <= 5,
        Lipinski.NumHAcceptors(mol) <= 10,
        -2 <= logp <= 5,
        Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10,
    ]
    return sum(int(r) for r in rules)


def _largest_fragment(mol):
    """Return the largest fragment of a molecule (handles disconnected SMILES)."""
    frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
    return max(frags, default=mol, key=lambda m: m.GetNumAtoms())


# ── Per-molecule metrics ──────────────────────────────────────────────────────

def compute_per_mol_metrics(smiles_list) -> list[dict]:
    """Compute QED, SA, LogP, Lipinski, and MW for each SMILES.

    Accepts either a plain ``list[str]`` (indices are assigned by position) or
    the ``list[tuple[int, str]]`` that ``load_smiles_from_file`` returns, in
    which case the given ``mol_idx`` is preserved so the output joins correctly
    against an eval CSV for the same file.

    Returns a list of dicts with keys:
        mol_idx, smiles, qed, sa, logp, lipinski, mw
    Properties are None when the SMILES cannot be parsed or sanitized.
    """
    indexed = [it if isinstance(it, tuple) else (i, it)
               for i, it in enumerate(smiles_list)]
    rows: list[dict] = []
    for i, smi in indexed:
        row: dict = {
            "mol_idx": i,
            "smiles": smi,
            "qed": None,
            "sa": None,
            "logp": None,
            "lipinski": None,
            "mw": None,
        }
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(row)
            continue
        try:
            Chem.SanitizeMol(mol)
            mol = _largest_fragment(mol)
            row["qed"] = round(QED.qed(mol), 4)
            row["sa"] = _normalized_sa(mol)
            row["logp"] = round(Crippen.MolLogP(mol), 4)
            row["lipinski"] = _lipinski_count(mol)
            row["mw"] = _molecular_weight(mol)
        except Exception:
            pass
        rows.append(row)
    return rows


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _std(xs) -> Optional[float]:
    """Population stdev — consistent with selectbench.metrics."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return pstdev(xs)


def _diversity(smiles_list: list[str], max_mols: int = 1000) -> Optional[float]:
    """Mean pairwise Tanimoto distance (1 - similarity) using RDKit fingerprints.

    Mirrors MoleculeProperties.calculate_diversity from DiffSBDD: pairwise over
    *all* molecules with no de-duplication, so duplicate/mode-collapsed outputs
    correctly lower the score. Capped at max_mols only to bound runtime (this
    cap never triggers on the summarizer path, which pre-caps to --cap_n_valid;
    it matters only for the standalone `quality` command on large inputs).
    """
    kept = [s for s in smiles_list if s]
    if len(kept) > max_mols:
        kept = kept[:max_mols]
    mols = [m for s in kept if (m := Chem.MolFromSmiles(s)) is not None]
    if len(mols) < 2:
        return 0.0
    fps = [Chem.RDKFingerprint(m) for m in mols]
    total, count = 0.0, 0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            total += 1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
            count += 1
    return round(total / count, 4) if count else None


def compute_aggregate_metrics(per_mol_rows: list[dict]) -> dict:
    """Aggregate per-molecule metrics from compute_per_mol_metrics output.

    Returns a dict with keys:
        n_total, n_scored,
        qed_mean, qed_std, sa_mean, sa_std,
        logp_mean, logp_std, lipinski_mean, lipinski_std,
        mw_mean, mw_std, diversity
    """
    def _as_float(v) -> Optional[float]:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    scored = [r for r in per_mol_rows if _as_float(r.get("qed")) is not None]

    qeds = [_as_float(r["qed"]) for r in scored]
    sas = [_as_float(r["sa"]) for r in scored]
    logps = [_as_float(r["logp"]) for r in scored]
    lipes = [_as_float(r["lipinski"]) for r in scored]
    mws = [_as_float(r.get("mw")) for r in scored]
    smiles = [r.get("smiles", "") for r in scored]

    def _fmt(v: Optional[float]) -> Optional[float]:
        return round(v, 4) if v is not None else None

    return {
        "n_total": len(per_mol_rows),
        "n_scored": len(scored),
        "qed_mean": _fmt(_mean(qeds)),
        "qed_std": _fmt(_std(qeds)),
        "sa_mean": _fmt(_mean(sas)),
        "sa_std": _fmt(_std(sas)),
        "logp_mean": _fmt(_mean(logps)),
        "logp_std": _fmt(_std(logps)),
        "lipinski_mean": _fmt(_mean(lipes)),
        "lipinski_std": _fmt(_std(lipes)),
        "mw_mean": _fmt(_mean(mws)),
        "mw_std": _fmt(_std(mws)),
        "diversity": _diversity(smiles),
    }


def compute_quality_from_smiles(smiles_list: list[str]) -> dict:
    """Convenience: per-mol → aggregate quality dict in one call.

    Used by the ``--quality`` summarize path, which has the ``smiles`` column
    from a raw results CSV in hand and only needs the aggregated row.
    """
    return compute_aggregate_metrics(compute_per_mol_metrics(smiles_list))
