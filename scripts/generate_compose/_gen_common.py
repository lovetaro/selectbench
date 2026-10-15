"""Shared helpers for per-baseline gen_<name>.py wrappers.

Unified contract:
- Wrappers accept --case_study, --poi, or --crossdocked and resolve paths
  automatically from the corresponding target CSV (no per-wrapper hardcoded
  dicts).
- Wrappers drive the baseline's native sampler via subprocess with cwd set
  to the baseline repo; callers do not need to `cd` into the baseline.
- Wrappers batch-accumulate until `target_count` valid connected molecules
  are collected, then concatenate to generated_samples/<name>/<target>.sdf
  (or generated_samples/<name>/crossdocked/<pdb>_<chain>.sdf for CrossDocked).
"""

import csv
import glob
import os
import random
from dataclasses import dataclass
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CASE_STUDIES_CSV = os.path.join(REPO_ROOT, "data", "case_studies", "case_studies.csv")
SDB_CSV = os.path.join(REPO_ROOT, "data", "standard_docking_benchmark",
                       "standard_docking_benchmark.csv")
CROSSDOCKED_CSV = os.path.join(REPO_ROOT, "data", "crossdocked_test_set",
                               "crossdocked_test_set_pairs.csv")
CROSSDOCKED_PROCESSED = os.path.join(REPO_ROOT, "data", "crossdocked_test_set",
                                     "processed")


@dataclass
class Target:
    type: str           # "case", "poi", or "crossdocked"
    name: str           # case_study/poi name, or <pdb>_<chain> for crossdocked
    pdb_id: str
    chain_id: str
    ligand_id: Optional[str]
    processed_dir: str
    pocket_pdb: str     # <pdb>_<chain>_pocket.pdb  (or crossdocked *_pocket10.box.pdb)
    receptor_pdb: str   # <pdb>_<chain>.pdb         (crossdocked: same as pocket_pdb)
    receptor_pdbqt: str # <pdb>_<chain>.pdbqt       (crossdocked: *_pocket10.pdbqt)
    ref_ligand_sdf: Optional[str]   # <lig>_ligand.sdf  (crossdocked: *_lig_tt_min_0.sdf)


def _load_case_studies():
    out = {}
    with open(CASE_STUDIES_CSV) as f:
        for r in csv.DictReader(f):
            out[r["case_name"]] = r
    return out


def _load_pois():
    out = {}
    with open(SDB_CSV) as f:
        for r in csv.DictReader(f):
            out[r["case_name"]] = r
    return out


def _load_crossdocked():
    """Return {<pdb>_<chain>: {pdb_id, chain_id, difficulty, processed_dir}}.

    Iterates every CSV row, recording each pdb1 / pdb2 pocket once. The same
    pocket may appear multiple times across pairs — later rows don't overwrite
    earlier entries.
    """
    out = {}
    with open(CROSSDOCKED_CSV) as f:
        for r in csv.DictReader(f):
            for k in ("pdb1", "pdb2"):
                pdb_id = r[f"{k}_id"]
                chain = r[f"{k}_chain_id"]
                name = f"{pdb_id}_{chain}"
                if name in out:
                    continue
                out[name] = {
                    "pdb_id": pdb_id,
                    "chain_id": chain,
                    "difficulty": r["difficulty"],
                    "processed_dir": os.path.join(CROSSDOCKED_PROCESSED, name),
                }
    return out


def resolve_target(name):
    """Resolve a case_study / PoI / CrossDocked name to a Target."""
    cases = _load_case_studies()
    pois = _load_pois()
    crossdocked = _load_crossdocked()

    if name in cases:
        row = cases[name]
        which = row["on_target"]  # "pdb1" or "pdb2"
        pdb_id = row[f"{which}_id"]
        chain_id = row[f"{which}_chain_id"]
        # case_studies.csv stores only pdb1's ligand id/smiles; use the single
        # *_ligand.sdf in processed/ regardless of which pdb is the on-target
        ligand_id = row.get("pdb1_ligand_id") or None
        processed_dir = os.path.join(REPO_ROOT, "data", "case_studies", name, "processed")
        type_ = "case"
    elif name in pois:
        row = pois[name]
        pdb_id = row["pdb_id"]
        chain_id = row["pdb_chain_id"]
        ligand_id = row.get("pdb_ligand_id") or None
        processed_dir = os.path.join(REPO_ROOT, "data", "standard_docking_benchmark",
                                     "processed", name)
        type_ = "poi"
    elif name in crossdocked:
        return _resolve_crossdocked(name, crossdocked[name])
    else:
        raise ValueError(
            f"target '{name}' not found in case_studies.csv, "
            f"standard_docking_benchmark.csv, or crossdocked_test_set_pairs.csv"
        )

    base = f"{pdb_id}_{chain_id}"
    receptor_pdb = os.path.join(processed_dir, f"{base}.pdb")
    pocket_pdb = os.path.join(processed_dir, f"{base}_pocket.pdb")
    receptor_pdbqt = os.path.join(processed_dir, f"{base}.pdbqt")

    # Prefer the CSV-named ligand; fall back to the single *_ligand.sdf if present
    ref_ligand = None
    if ligand_id:
        cand = os.path.join(processed_dir, f"{ligand_id}_ligand.sdf")
        if os.path.isfile(cand):
            ref_ligand = cand
    if ref_ligand is None:
        matches = sorted(glob.glob(os.path.join(processed_dir, "*_ligand.sdf")))
        if matches:
            ref_ligand = matches[0]

    return Target(type=type_, name=name, pdb_id=pdb_id, chain_id=chain_id,
                  ligand_id=ligand_id, processed_dir=processed_dir,
                  pocket_pdb=pocket_pdb, receptor_pdb=receptor_pdb,
                  receptor_pdbqt=receptor_pdbqt, ref_ligand_sdf=ref_ligand)


def _resolve_crossdocked(name, row):
    """Build a Target for a CrossDocked pocket.

    Everything comes from ``processed/<name>/``. Use ``*_pocket10.pdb`` for the
    protein pocket (10 Å crop, no ligand) — NOT its ``*_pocket10.box.pdb``
    sibling, which is a 9-atom Vina-box placeholder (8 BOX corners + 1 center,
    dummy Ne/Xe atoms), not a usable pocket. The ``*_pocket10.pdb`` glob
    excludes the ``.box.pdb`` file by suffix; the file is byte-identical to the
    copy under ``raw/<uniprot>/`` (which we no longer depend on). The ligand
    SDF varies by suffix across the test set: 63 targets use
    ``*_lig_tt_min_0.sdf`` and 37 use ``*_lig_*tt_docked_N.sdf`` — both match
    ``*_lig*.sdf``.
    """
    processed_dir = row["processed_dir"]
    if not os.path.isdir(processed_dir):
        raise ValueError(f"CrossDocked processed dir missing: {processed_dir}")

    def _one(pattern, root, required=True):
        matches = sorted(glob.glob(os.path.join(root, pattern)))
        if not matches:
            if required:
                raise ValueError(
                    f"No {pattern} in {root} (crossdocked target {name})")
            return None
        return matches[0]

    receptor_pdbqt = _one("*_pocket10.pdbqt", processed_dir)
    ref_ligand     = _one("*_lig*.sdf",       processed_dir)
    # The protein pocket (10 Å crop, no ligand). "*_pocket10.pdb" excludes the
    # "*_pocket10.box.pdb" box placeholder by suffix.
    pocket_pdb     = _one("*_pocket10.pdb",   processed_dir)

    return Target(
        type="crossdocked", name=name,
        pdb_id=row["pdb_id"], chain_id=row["chain_id"],
        ligand_id=None, processed_dir=processed_dir,
        pocket_pdb=pocket_pdb,
        receptor_pdb=pocket_pdb,    # no full receptor — cropped pocket stands in
        receptor_pdbqt=receptor_pdbqt,
        ref_ligand_sdf=ref_ligand,
    )


def list_crossdocked_targets(difficulty, n_samples=None, seed=0):
    """List unique on-target pockets at the given difficulty.

    Args:
        difficulty: 'easy', 'medium', 'hard', or 'all'.
        n_samples: If set, subsample to this many (random with `seed`).
                   Ignored when None.
        seed: RNG seed for subsampling.

    Returns:
        Sorted list of <pdb>_<chain> names (deterministic before subsampling;
        subsample is random-seeded, then re-sorted).
    """
    out = set()
    with open(CROSSDOCKED_CSV) as f:
        for r in csv.DictReader(f):
            if difficulty != "all" and r["difficulty"] != difficulty:
                continue
            k = "pdb1" if r["on_target"] == "pdb1" else "pdb2"
            out.add(f"{r[f'{k}_id']}_{r[f'{k}_chain_id']}")
    names = sorted(out)
    if n_samples is not None and n_samples < len(names):
        rng = random.Random(seed)
        names = sorted(rng.sample(names, n_samples))
    return names


def list_crossdocked_pairs(difficulty, n_samples=None, seed=0):
    """List (case_id, on_name, off_name) pairs at the given difficulty.

    Args are as in ``list_crossdocked_targets``. Each pair corresponds to
    exactly one CSV row (oriented — pdb1 as on vs pdb2 as on are two rows).
    """
    pairs = []
    with open(CROSSDOCKED_CSV) as f:
        for r in csv.DictReader(f):
            if difficulty != "all" and r["difficulty"] != difficulty:
                continue
            on_k  = "pdb1" if r["on_target"] == "pdb1" else "pdb2"
            off_k = "pdb2" if r["on_target"] == "pdb1" else "pdb1"
            on_name  = f"{r[f'{on_k}_id']}_{r[f'{on_k}_chain_id']}"
            off_name = f"{r[f'{off_k}_id']}_{r[f'{off_k}_chain_id']}"
            pairs.append((int(r["case_id"]), on_name, off_name))
    pairs.sort()
    if n_samples is not None and n_samples < len(pairs):
        rng = random.Random(seed)
        pairs = sorted(rng.sample(pairs, n_samples))
    return pairs


def crossdocked_row_by_case_id(case_id):
    """Return the CSV row dict for a given case_id (int)."""
    with open(CROSSDOCKED_CSV) as f:
        for r in csv.DictReader(f):
            if int(r["case_id"]) == case_id:
                return r
    raise ValueError(f"CrossDocked case_id {case_id} not found")


def resolve_compose_pair(case_study=None, crossdocked_case=None):
    """Return (pocket_on_pdb_path, pocket_off_pdb_path, label, pair_tag).

    Used by compose wrappers that need both pockets of a pair.

    Args:
        case_study: name from case_studies.csv (e.g. "parp"). Uses the CSV's
            ``on_target`` to pick on vs off.
        crossdocked_case: integer case_id from the CrossDocked CSV.

    Returns:
        (on_pdb, off_pdb, label, pair_tag) where:
        - label: short human name for logs (case-study name, or "xd<case_id>")
        - pair_tag: filename-safe string identifying the pair (case-study
          name, or "<on>_vs_<off>" for crossdocked — caller is expected to
          append the difficulty suffix).
    """
    if (case_study is None) == (crossdocked_case is None):
        raise ValueError("resolve_compose_pair needs exactly one of "
                         "case_study / crossdocked_case")

    if case_study is not None:
        cases = _load_case_studies()
        if case_study not in cases:
            raise ValueError(f"case_study {case_study!r} not in case_studies.csv")
        row = cases[case_study]
        on_k  = "pdb1" if row["on_target"] == "pdb1" else "pdb2"
        off_k = "pdb2" if row["on_target"] == "pdb1" else "pdb1"
        processed = os.path.join(REPO_ROOT, "data", "case_studies", case_study, "processed")
        on_pdb  = os.path.join(processed,
            f"{row[f'{on_k}_id']}_{row[f'{on_k}_chain_id']}_pocket.pdb")
        off_pdb = os.path.join(processed,
            f"{row[f'{off_k}_id']}_{row[f'{off_k}_chain_id']}_pocket.pdb")
        return on_pdb, off_pdb, case_study, case_study

    # CrossDocked pair
    row = crossdocked_row_by_case_id(crossdocked_case)
    on_k  = "pdb1" if row["on_target"] == "pdb1" else "pdb2"
    off_k = "pdb2" if row["on_target"] == "pdb1" else "pdb1"
    on_name  = f"{row[f'{on_k}_id']}_{row[f'{on_k}_chain_id']}"
    off_name = f"{row[f'{off_k}_id']}_{row[f'{off_k}_chain_id']}"

    def _pocket(name):
        # Use processed/<name>/*_pocket10.pdb (the real protein pocket). Its
        # *_pocket10.box.pdb sibling is just a 9-atom Vina-box placeholder;
        # the "*_pocket10.pdb" glob excludes it by suffix.
        proc = os.path.join(REPO_ROOT, "data", "crossdocked_test_set",
                            "processed", name)
        hits = sorted(glob.glob(os.path.join(proc, "*_pocket10.pdb")))
        if not hits:
            raise ValueError(f"No *_pocket10.pdb for {name} under {proc}")
        return hits[0]

    on_pdb = _pocket(on_name)
    off_pdb = _pocket(off_name)
    label = f"xd{crossdocked_case}"
    pair_tag = f"{on_name}_vs_{off_name}"
    return on_pdb, off_pdb, label, pair_tag


def filter_valid_connected(sdf_path):
    """Return list of (mol, smiles) for RDKit-valid, single-component mols in sdf_path."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = []
    if not os.path.isfile(sdf_path) or os.path.getsize(sdf_path) == 0:
        return out
    sup = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
    for m in sup:
        if m is None:
            continue
        try:
            smi = Chem.MolToSmiles(m)
        except Exception:
            continue
        if not smi or "." in smi:
            continue
        out.append((m, smi))
    return out


def default_output_path(target, method_dir, ext=".sdf"):
    """Default wrapper output path for a target.

    Outputs are split by eval type so downstream tools can fan out per kind:
        case_study  → <method_dir>/cs/<name>.<ext>
        PoI         → <method_dir>/safety/<name>.<ext>
        crossdocked → <method_dir>/crossdocked/<name>.<ext>
    """
    subdir = {"case": "cs", "poi": "safety", "crossdocked": "crossdocked"}[target.type]
    return os.path.join(method_dir, subdir, f"{target.name}{ext}")


def concat_to_sdf(mols, out_path, max_n=None):
    """Write mols (list of RDKit Mol) to out_path as a multi-mol SDF, capped at max_n."""
    from rdkit import Chem
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    w = Chem.SDWriter(out_path)
    n = 0
    for m in mols:
        if max_n is not None and n >= max_n:
            break
        w.write(m)
        n += 1
    w.close()
    return n
