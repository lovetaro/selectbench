# selectbench/cli.py
import os
import csv
import glob
import hashlib
import math
import click
from multiprocessing import cpu_count, get_context
from tqdm import tqdm

from rdkit import Chem


def _safe_nproc(cap: int = 8) -> int:
    """SLURM-aware default for --nproc. Never bigger than ``cap``.

    Raw ``cpu_count()`` reports the host's total logical CPUs, which
    over-subscribes on shared clusters when the job was allocated fewer.
    Prefer ``$SLURM_CPUS_PER_TASK``, else the actually-affinable core count
    on Linux, else fall back to ``cpu_count()``. Cap at 8 by default to
    protect other users' jobs; users who genuinely want more pass ``--nproc``.
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            return min(cap, int(slurm))
        except ValueError:
            pass
    if hasattr(os, "sched_getaffinity"):
        return min(cap, len(os.sched_getaffinity(0)))
    return min(cap, cpu_count())

from selectbench.data.ligand_prep import prepare_ligand
from selectbench.data.dock import run_vina, run_vina_safe, _dock_single_target, _prep_ligand_worker, _dock_pdbqt_worker, NoDaemonPool
from selectbench.metrics import EXCLUDED_SAFETY_TARGETS

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CS_DATA_DIR = os.path.join(PACKAGE_ROOT, "data", "case_studies")
ST_DATA_DIR = os.path.join(PACKAGE_ROOT, "data", "safety_targets")
SDB_DATA_DIR = os.path.join(PACKAGE_ROOT, "data", "standard_docking_benchmark")
CD_DATA_DIR = os.path.join(PACKAGE_ROOT, "data", "crossdocked_test_set")

# Per-suite Vina search budgets (the paper protocol). The check-* commands
# reuse these so a one-molecule sanity check is comparable to the benchmark
# rather than silently running at vina's own default of 20.
EXHAUSTIVENESS_CASE = 20
EXHAUSTIVENESS_CROSSDOCKED = 10
EXHAUSTIVENESS_SAFETY = 1



def load_case_studies(csv_path):
    cases = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases[row["case_name"]] = row
    return cases


def load_safety_cases(csv_path):
    """Load safety_targets.csv and drop entries in EXCLUDED_SAFETY_TARGETS
    (htr1b_agonist is the same protein as the 5ht1b PoI — including it as
    an off-target would double-count the on-target)."""
    cases = load_case_studies(csv_path)
    return {k: v for k, v in cases.items() if k not in EXCLUDED_SAFETY_TARGETS}


@click.group()
def cli():
    pass


@cli.command()
@click.option("--case_id", required=True, type=int, help="Case ID")
@click.option("--smiles", required=True, help="Input ligand SMILES")
@click.option(
    "--save_poses_dir",
    type=click.Path(),
    default=None,
    help="Directory to save docked poses (will be created if it doesn't exist)",
)
@click.option("--exhaustiveness", default=EXHAUSTIVENESS_CROSSDOCKED, show_default=True,
              help="Vina search budget. Defaults to the CrossDocked protocol so "
                   "this check is comparable to `evaluate-xl`.")
def check_crossdocked(case_id, smiles, save_poses_dir, exhaustiveness):
    """
    Run docking of a SMILES ligand on both receptors of a crossdocked test set case.

    Uses the same search budget as ``evaluate-xl`` so the numbers are directly
    comparable; pass ``--exhaustiveness`` to override.
    """
    csv_path = os.path.join(CD_DATA_DIR, "crossdocked_test_set_pairs.csv")
    row = None
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if int(r["case_id"]) == case_id:
                row = r
                break

    if row is None:
        raise click.BadParameter(f"case_id {case_id} not found in {csv_path}")

    # --- Receptors & boxes ---
    pdb1_key = f"{row['pdb1_id']}_{row['pdb1_chain_id']}"
    pdb2_key = f"{row['pdb2_id']}_{row['pdb2_chain_id']}"

    def find_receptor_files(pdb_key):
        processed_dir = os.path.join(CD_DATA_DIR, "processed", pdb_key)
        pdbqt_matches = glob.glob(os.path.join(processed_dir, "*_pocket10.pdbqt"))
        box_matches = glob.glob(os.path.join(processed_dir, "*_pocket10.box.txt"))
        if not pdbqt_matches:
            raise FileNotFoundError(f"No receptor .pdbqt found in {processed_dir}")
        if not box_matches:
            raise FileNotFoundError(f"No box file found in {processed_dir}")
        return pdbqt_matches[0], box_matches[0]

    receptor1, box_file1 = find_receptor_files(pdb1_key)
    receptor2, box_file2 = find_receptor_files(pdb2_key)

    # --- Prepare ligand ---
    ligand_pdbqt = prepare_ligand(smiles)

    # --- Dock ---
    if save_poses_dir:
        os.makedirs(save_poses_dir, exist_ok=True)

    docked1 = os.path.join(save_poses_dir, f"docked_{pdb1_key}.pdbqt") if save_poses_dir else None
    docked2 = os.path.join(save_poses_dir, f"docked_{pdb2_key}.pdbqt") if save_poses_dir else None

    out1 = run_vina(ligand_pdbqt, receptor1, box_file1, docked1,
                    exhaustiveness=exhaustiveness)
    out2 = run_vina(ligand_pdbqt, receptor2, box_file2, docked2,
                    exhaustiveness=exhaustiveness)

    print(out1)
    print(out2)
    print("Docking completed.")


def _dock_pair_worker(args):
    """Multiprocessing worker: dock one molecule against both receptors of a case study.

    Tuple layout: (mol_idx, smiles, r1, b1, r2, b2, on_target,
                   exhaustiveness [, timeout])
    Optional 9th element is a per-call timeout override for run_vina_safe
    (used by the fallback retry path).

    Writes only the per-mol primitives ``mol_idx``, ``smiles``,
    ``score_on_target``, ``score_off_target``, ``delta``. Z-scores and
    aggregate metrics are computed at summary time (see
    ``scripts/summarize/cross_method.py``).
    """
    timeout = None
    if len(args) == 9:
        (mol_idx, smiles, receptor1, box_file1, receptor2, box_file2,
         on_target, exhaustiveness, timeout) = args
    else:
        (mol_idx, smiles, receptor1, box_file1, receptor2, box_file2,
         on_target, exhaustiveness) = args
    dock_kwargs = {"exhaustiveness": exhaustiveness}
    if timeout is not None:
        dock_kwargs["timeout"] = timeout
    try:
        ligand_pdbqt = prepare_ligand(smiles)
        out1 = run_vina_safe(ligand_pdbqt, receptor1, box_file1, **dock_kwargs)
        out2 = run_vina_safe(ligand_pdbqt, receptor2, box_file2, **dock_kwargs)
        score1 = float(out1[0][0])
        score2 = float(out2[0][0])
        score_on  = score1 if on_target == "pdb1" else score2
        score_off = score2 if on_target == "pdb1" else score1
        delta = score_on - score_off
        row = {"mol_idx": mol_idx, "smiles": smiles,
               "score_on_target": score_on, "score_off_target": score_off, "delta": delta}
    except Exception as e:
        row = {"mol_idx": mol_idx, "smiles": smiles,
               "score_on_target": None, "score_off_target": None, "delta": None,
               "_error": str(e)}
    return row


FALLBACK_EXHAUSTIVENESS_DEFAULT = 10  # legacy fallback used by older callers
FALLBACK_EXHAUSTIVENESS_CASE = 8
FALLBACK_EXHAUSTIVENESS_SAFETY = 1
FALLBACK_EXHAUSTIVENESS_CROSSDOCKED = 4
FALLBACK_TIMEOUT_DEFAULT = 1200


def _retry_failed_cs_rows(rows, build_task, label, exhaustiveness, timeout):
    """Re-dock rows with ``delta is None`` serially (nproc=1) using a lower
    exhaustiveness and higher per-mol timeout. Replaces the row in place only
    if the retry produces a valid delta.

    ``build_task(mol_idx, smiles)`` returns the 12-tuple args for
    ``_dock_pair_worker`` with exhaustiveness and timeout baked in.

    Recovered rows are stamped with the exhaustiveness that actually produced
    them, because it differs from the first pass (20 -> 8 for case studies,
    10 -> 4 for CrossDocked). Whether a molecule times out depends on host load,
    not on the seed, so without that column two runs of the same command can
    disagree on a row with nothing in the CSV to explain why.
    """
    failed_idx = [i for i, r in enumerate(rows) if r.get("delta") is None]
    if not failed_idx:
        return rows
    print(f"\n[fallback:{label}] retrying {len(failed_idx)} failed mol(s) "
          f"(nproc=1, exhaustiveness={exhaustiveness}, timeout={timeout}s)")
    n_recovered = 0
    for i in tqdm(failed_idx, desc=f"Fallback [{label}]", unit="mol"):
        task = build_task(rows[i]["mol_idx"], rows[i]["smiles"])
        new_row = _dock_pair_worker(task)
        err = new_row.pop("_error", None)
        if new_row.get("delta") is not None:
            new_row["exhaustiveness"] = exhaustiveness
            rows[i] = new_row
            n_recovered += 1
        elif err:
            tqdm.write(f"  [{new_row['mol_idx']}] still failed: {err}")
    print(f"[fallback:{label}] recovered {n_recovered}/{len(failed_idx)} rows")
    return rows


def _retry_failed_safety_dockings(dock_map, prep_map, receptor_box_by_target,
                                  exhaustiveness, timeout):
    """Re-dock (mol_idx, target_label) pairs whose score is ``None`` in
    ``dock_map``. Serial (nproc=1), lower exhaustiveness, higher timeout.
    Updates ``dock_map`` in place; skips mols whose ligand prep failed.

    ``receptor_box_by_target`` maps target_label -> (receptor_path, box_path).
    """
    failed_tasks = []
    for mol_idx, target_scores in dock_map.items():
        prep_r = prep_map.get(mol_idx)
        if not prep_r or prep_r.get("error") or not prep_r.get("pdbqt"):
            continue
        pdbqt = prep_r["pdbqt"]
        for target_label, score in target_scores.items():
            if score is None and target_label in receptor_box_by_target:
                receptor, box = receptor_box_by_target[target_label]
                failed_tasks.append(
                    (mol_idx, pdbqt, receptor, box, target_label, exhaustiveness, timeout)
                )
    if not failed_tasks:
        return dock_map
    print(f"\n[fallback:safety] retrying {len(failed_tasks)} failed (mol, target) "
          f"docking(s) (nproc=1, exhaustiveness={exhaustiveness}, timeout={timeout}s)")
    n_recovered = 0
    for task in tqdm(failed_tasks, desc="Fallback [safety]", unit="task"):
        result = _dock_pdbqt_worker(task)
        if result["score"] is not None:
            dock_map[result["mol_idx"]][result["target_label"]] = result["score"]
            n_recovered += 1
    print(f"[fallback:safety] recovered {n_recovered}/{len(failed_tasks)} dockings")
    return dock_map


def _dock_molecules(case, case_study, smiles_list, nproc=1, exhaustiveness=20,
                   fallback=True, fallback_exhaustiveness=FALLBACK_EXHAUSTIVENESS_DEFAULT,
                   fallback_timeout=FALLBACK_TIMEOUT_DEFAULT):
    """
    Dock a list of (idx, smiles) tuples against both receptors of a case study.
    Returns a list of per-mol rows with: mol_idx, smiles, score_on_target,
    score_off_target, delta.

    Z-score columns and aggregate metrics (selectivity_rate, SR_1kcal,
    top-K deltas, etc.) are computed on demand by
    ``scripts/summarize/cross_method.py``, not embedded in the CSV.
    """
    processed_dir = os.path.join(CS_DATA_DIR, case_study, "processed")

    pdb1_id = case["pdb1_id"]
    pdb1_chain = case["pdb1_chain_id"]
    pdb2_id = case["pdb2_id"]
    pdb2_chain = case["pdb2_chain_id"]
    on_target = case["on_target"]

    receptor1 = os.path.join(processed_dir, f"{pdb1_id}_{pdb1_chain}.pdbqt")
    receptor2 = os.path.join(processed_dir, f"{pdb2_id}_{pdb2_chain}.pdbqt")
    box_file1 = os.path.join(processed_dir, f"{pdb1_id}_{pdb1_chain}.box.txt")
    box_file2 = os.path.join(processed_dir, f"{pdb2_id}_{pdb2_chain}.box.txt")

    if not all(os.path.exists(p) for p in [receptor1, receptor2, box_file1, box_file2]):
        raise FileNotFoundError(f"Processed receptor or docking box not found for {case_study}")

    on_key  = f"{pdb1_id}_{pdb1_chain}" if on_target == "pdb1" else f"{pdb2_id}_{pdb2_chain}"

    print(f"[{case_study}] {len(smiles_list)} molecules | on-target: {on_key} | nproc={nproc}")

    tasks = [
        (i, smiles, receptor1, box_file1, receptor2, box_file2,
         on_target, exhaustiveness)
        for i, smiles in smiles_list
    ]

    with NoDaemonPool(processes=nproc) as pool:
        rows = list(tqdm(
            pool.imap_unordered(_dock_pair_worker, tasks),
            total=len(tasks), desc=f"Docking [{case_study}]", unit="mol",
        ))
    # imap_unordered yields in completion order, so sort to make the emitted
    # CSV byte-reproducible across runs (metrics key off mol_idx and are
    # unaffected, but two identical runs should be diffable).
    rows.sort(key=lambda r: r["mol_idx"])
    # Stamp the first-pass budget; _retry_failed_cs_rows overwrites it on any
    # row it recovers, so each score records the exhaustiveness that produced it.
    for _r in rows:
        _r["exhaustiveness"] = exhaustiveness

    for row in rows:
        err = row.pop("_error", None)
        if err:
            print(f"  [{row['mol_idx']}] Failed ({row['smiles'][:40]}...): {err}")
        elif row["delta"] is not None:
            print(f"  [{row['mol_idx']}] {row['smiles'][:50]}  "
                  f"on={row['score_on_target']:.2f}  off={row['score_off_target']:.2f}  "
                  f"delta={row['delta']:.2f}")

    if fallback:
        def _build_task(mol_idx, smiles):
            return (mol_idx, smiles, receptor1, box_file1, receptor2, box_file2,
                    on_target, fallback_exhaustiveness, fallback_timeout)
        rows = _retry_failed_cs_rows(rows, _build_task, case_study,
                                     fallback_exhaustiveness, fallback_timeout)

    # Raw per-mol CSV only — aggregate metrics are computed on demand by
    # scripts/summarize/. Print a one-line console summary for convenience.
    scored = [r for r in rows if r["delta"] is not None]
    if scored:
        from selectbench.metrics import cs_metrics
        m = cs_metrics(rows)
        print(f"\n  [{case_study}] n={m['n_valid']}/{m['n_total']}  "
              f"on={m['score_on_target']:.2f}  off={m['score_off_target']:.2f}  "
              f"Δ={m['delta']:.2f}  SR={m['selectivity_rate']:.2f}  "
              f"SR_1kcal={m['selectivity_rate_1kcal']:.2f}")
    return rows


def _dock_molecules_crossdocked(row, case_id, smiles_list,
                                nproc=1, exhaustiveness=20,
                                fallback=True,
                                fallback_exhaustiveness=FALLBACK_EXHAUSTIVENESS_DEFAULT,
                                fallback_timeout=FALLBACK_TIMEOUT_DEFAULT):
    """Dock an iterable of (idx, SMILES) against both pockets of a CrossDocked
    pair identified by a CSV row (from ``crossdocked_test_set_pairs.csv``).

    Returns per-mol rows with the same shape as ``_dock_molecules``.
    """
    pdb1_key = f"{row['pdb1_id']}_{row['pdb1_chain_id']}"
    pdb2_key = f"{row['pdb2_id']}_{row['pdb2_chain_id']}"
    on_target = row["on_target"]  # 'pdb1' or 'pdb2'

    def _find(pdb_key, pattern):
        hits = glob.glob(os.path.join(CD_DATA_DIR, "processed", pdb_key, pattern))
        if not hits:
            raise FileNotFoundError(
                f"No {pattern} in {CD_DATA_DIR}/processed/{pdb_key}")
        return hits[0]

    receptor1 = _find(pdb1_key, "*_pocket10.pdbqt")
    receptor2 = _find(pdb2_key, "*_pocket10.pdbqt")
    box_file1 = _find(pdb1_key, "*_pocket10.box.txt")
    box_file2 = _find(pdb2_key, "*_pocket10.box.txt")

    on_key  = pdb1_key if on_target == "pdb1" else pdb2_key
    off_key = pdb2_key if on_target == "pdb1" else pdb1_key

    label = f"xd{case_id}"
    print(f"[{label}] {len(smiles_list)} molecules | on-target: {on_key} | "
          f"off: {off_key} | nproc={nproc}")

    tasks = [
        (i, smiles, receptor1, box_file1, receptor2, box_file2,
         on_target, exhaustiveness)
        for i, smiles in smiles_list
    ]

    with NoDaemonPool(processes=nproc) as pool:
        rows = list(tqdm(
            pool.imap_unordered(_dock_pair_worker, tasks),
            total=len(tasks), desc=f"Docking [{label}]", unit="mol",
        ))
    rows.sort(key=lambda r: r["mol_idx"])
    # Stamp the first-pass budget; _retry_failed_cs_rows overwrites it on any
    # row it recovers, so each score records the exhaustiveness that produced it.
    for _r in rows:
        _r["exhaustiveness"] = exhaustiveness

    for r in rows:
        err = r.pop("_error", None)
        if err:
            print(f"  [{r['mol_idx']}] Failed ({r['smiles'][:40]}...): {err}")
        elif r["delta"] is not None:
            print(f"  [{r['mol_idx']}] {r['smiles'][:50]}  "
                  f"on={r['score_on_target']:.2f}  off={r['score_off_target']:.2f}  "
                  f"delta={r['delta']:.2f}")

    if fallback:
        def _build_task(mol_idx, smiles):
            return (mol_idx, smiles, receptor1, box_file1, receptor2, box_file2,
                    on_target, fallback_exhaustiveness, fallback_timeout)
        rows = _retry_failed_cs_rows(rows, _build_task, label,
                                     fallback_exhaustiveness, fallback_timeout)

    scored = [r for r in rows if r["delta"] is not None]
    if scored:
        from selectbench.metrics import cs_metrics
        m = cs_metrics(rows)
        print(f"\n  [{label}] n={m['n_valid']}/{m['n_total']}  "
              f"on={m['score_on_target']:.2f}  off={m['score_off_target']:.2f}  "
              f"Δ={m['delta']:.2f}  SR={m['selectivity_rate']:.2f}  "
              f"SR_1kcal={m['selectivity_rate_1kcal']:.2f}")
    return rows


def _dock_sdf(case, case_study, sdf_file, nproc=1, exhaustiveness=20, **kwargs):
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=True)
    smiles_list = [(i, Chem.MolToSmiles(mol)) for i, mol in enumerate(suppl) if mol is not None]
    return _dock_molecules(case, case_study, smiles_list,
                           nproc=nproc, exhaustiveness=exhaustiveness, **kwargs)


def _dock_smiles_file(case, case_study, smiles_file, nproc=1, exhaustiveness=20, **kwargs):
    with open(smiles_file) as f:
        smiles_list = [(i, line.strip()) for i, line in enumerate(f) if line.strip()]
    return _dock_molecules(case, case_study, smiles_list,
                           nproc=nproc, exhaustiveness=exhaustiveness, **kwargs)


@cli.command("evaluate-case")
@click.option("--case_study", required=True, help="Name of the case study")
@click.option("--sdf_file", type=click.Path(exists=True), default=None, help="SDF file with generated molecules")
@click.option("--smiles_file", type=click.Path(exists=True), default=None, help="Text file with one SMILES per line")
@click.option("--nproc", default=_safe_nproc(), show_default=True, help="Number of parallel docking workers")
@click.option("--exhaustiveness", default=EXHAUSTIVENESS_CASE, show_default=True,
              help="Vina exhaustiveness (paper protocol for case studies: 20).")
@click.option("--fallback/--no-fallback", default=True, show_default=True,
              help="Disable the post-run fallback retry (nproc=1, lower "
                   "exhaustiveness, higher timeout) on failed dockings. "
                   "Fallback is ON by default.")
@click.option("--fallback_exhaustiveness", default=FALLBACK_EXHAUSTIVENESS_CASE,
              show_default=True, help="Vina exhaustiveness for fallback retries.")
@click.option("--fallback_timeout", default=FALLBACK_TIMEOUT_DEFAULT, show_default=True,
              help="Per-molecule timeout (seconds) for fallback retries.")
@click.option("--output", type=click.Path(), default=None,
              help="Output CSV path. Default: results/evaluate-case/<case_study>.csv.")
def evaluate_case(case_study, sdf_file, smiles_file, nproc, exhaustiveness,
                  fallback, fallback_exhaustiveness, fallback_timeout, output):
    """
    Evaluate molecules from an SDF or SMILES file against a single case study.
    Outputs a per-mol CSV with Vina scores and selectivity delta. Z-scores
    and aggregate metrics are produced by ``scripts/summarize/cross_method.py``.
    """
    if not sdf_file and not smiles_file:
        raise click.UsageError("Provide either --sdf_file or --smiles_file.")
    if sdf_file and smiles_file:
        raise click.UsageError("Provide only one of --sdf_file or --smiles_file.")

    cases = load_case_studies(os.path.join(CS_DATA_DIR, "case_studies.csv"))
    if case_study not in cases:
        raise click.BadParameter(f"Unknown case study: {case_study}", param_hint="--case_study")
    if not output:
        output = os.path.join("results", "evaluate-case", f"{case_study}.csv")

    common = {"nproc": nproc, "exhaustiveness": exhaustiveness,
              "fallback": fallback, "fallback_exhaustiveness": fallback_exhaustiveness,
              "fallback_timeout": fallback_timeout}
    if sdf_file:
        rows = _dock_sdf(cases[case_study], case_study, sdf_file, **common)
    else:
        rows = _dock_smiles_file(cases[case_study], case_study, smiles_file, **common)

    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fieldnames = list(dict.fromkeys(k for row in rows for k in row.keys()))
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {output}")


@cli.command("evaluate-xl")
@click.option("--case_id", required=True, type=int,
              help="CrossDocked CSV row id (from crossdocked_test_set_pairs.csv)")
@click.option("--sdf_file", type=click.Path(exists=True), default=None,
              help="SDF file with generated molecules")
@click.option("--smiles_file", type=click.Path(exists=True), default=None,
              help="Text file with one SMILES per line")
@click.option("--nproc", default=_safe_nproc(), show_default=True,
              help="Number of parallel docking workers")
@click.option("--exhaustiveness", default=EXHAUSTIVENESS_CROSSDOCKED, show_default=True,
              help="Vina exhaustiveness (paper protocol for CrossDocked: 10).")
@click.option("--fallback/--no-fallback", default=True, show_default=True,
              help="Disable the post-run fallback retry (nproc=1, lower "
                   "exhaustiveness, higher timeout) on failed dockings. "
                   "Fallback is ON by default.")
@click.option("--fallback_exhaustiveness", default=FALLBACK_EXHAUSTIVENESS_CROSSDOCKED,
              show_default=True, help="Vina exhaustiveness for fallback retries.")
@click.option("--fallback_timeout", default=FALLBACK_TIMEOUT_DEFAULT, show_default=True,
              help="Per-molecule timeout (seconds) for fallback retries.")
@click.option("--output", type=click.Path(), default=None,
              help="Output CSV path. Default: results/evaluate-xl/case<case_id>.csv.")
def evaluate_crossdocked(case_id, sdf_file, smiles_file,
                         nproc, exhaustiveness,
                         fallback, fallback_exhaustiveness, fallback_timeout,
                         output):
    """Evaluate molecules against a CrossDocked pair identified by case_id.

    Pockets are resolved from ``data/crossdocked_test_set/crossdocked_test_set_pairs.csv``
    (row with the given case_id) and ``processed/<pdb>_<chain>/``. Outputs a
    per-mol CSV; Z-scores and aggregates are summary-time only.
    """
    if not sdf_file and not smiles_file:
        raise click.UsageError("Provide either --sdf_file or --smiles_file.")
    if sdf_file and smiles_file:
        raise click.UsageError("Provide only one of --sdf_file or --smiles_file.")

    csv_path = os.path.join(CD_DATA_DIR, "crossdocked_test_set_pairs.csv")
    row = None
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if int(r["case_id"]) == case_id:
                row = r
                break
    if row is None:
        raise click.BadParameter(f"case_id {case_id} not found in {csv_path}", param_hint="--case_id")
    if not output:
        output = os.path.join("results", "evaluate-xl", f"case{case_id}.csv")

    if sdf_file:
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=True)
        smiles_list = [(i, Chem.MolToSmiles(mol))
                       for i, mol in enumerate(suppl) if mol is not None]
    else:
        with open(smiles_file) as f:
            smiles_list = [(i, line.strip()) for i, line in enumerate(f) if line.strip()]

    rows = _dock_molecules_crossdocked(row, case_id, smiles_list,
                                       nproc=nproc,
                                       exhaustiveness=exhaustiveness,
                                       fallback=fallback,
                                       fallback_exhaustiveness=fallback_exhaustiveness,
                                       fallback_timeout=fallback_timeout)

    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fieldnames = list(dict.fromkeys(k for row in rows for k in row.keys()))
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {output}")


# ---------------------------------------------------------------------------
# check-safety: single SMILES vs all safety targets (parallel)
# ---------------------------------------------------------------------------

@cli.command("check-safety")
@click.option("--smiles", required=True, help="Input ligand SMILES")
@click.option("--priority_targets", is_flag=True, help="Only consider priority targets")
@click.option("--save_poses_dir", type=click.Path(), default=None,
              help="Directory to save docked poses (created if absent)")
@click.option("--nproc", default=_safe_nproc(), show_default=True,
              help="Number of processes to use")
@click.option("--exhaustiveness", default=EXHAUSTIVENESS_SAFETY, show_default=True,
              help="Vina search budget. Defaults to the safety-panel protocol so "
                   "this check is comparable to `evaluate-safety`.")
def check_safety(smiles, priority_targets, save_poses_dir, nproc, exhaustiveness):
    """
    Run docking of a SMILES ligand on the safety targets.

    Uses the same search budget as ``evaluate-safety`` so the numbers are
    directly comparable; pass ``--exhaustiveness`` to override.
    """
    csv_path = os.path.join(ST_DATA_DIR, "safety_targets.csv")
    cases = load_safety_cases(csv_path)

    if priority_targets:
        cases = {k: v for k, v in cases.items() if int(v["priority_target"]) == 1}

    ligand_pdbqt = prepare_ligand(smiles)

    print(f"Running docking on {len(cases)} targets using {nproc} processes.")

    if save_poses_dir:
        os.makedirs(save_poses_dir, exist_ok=True)

    tasks = [
        (case_row, ST_DATA_DIR, save_poses_dir, ligand_pdbqt, exhaustiveness)
        for case_row in cases.values()
    ]

    with NoDaemonPool(processes=nproc) as pool:
        results = pool.map(_dock_single_target, tasks)

    for r in results:
        if r["error"]:
            print(f"[{r['case_name']}] ERROR: {r['error']}")
        else:
            print(f"[{r['case_name']}] score = {r['score']:.3f}")

    print("Docking completed.")


# ---------------------------------------------------------------------------
# evaluate-safety: dock many molecules vs all safety targets, output CSV
# ---------------------------------------------------------------------------

@cli.command("evaluate-safety")
@click.option(
    "--poi",
    default=None,
    help="Protein of interest from standard_docking_benchmark.csv (e.g. parp1, jak2, braf, fa7, 5ht1b). "
         "Mutually exclusive with --on_target_receptor / --on_target_box.",
)
@click.option(
    "--on_target_receptor",
    type=click.Path(exists=True),
    default=None,
    help="Path to custom on-target receptor .pdbqt file. Requires --on_target_box. "
         "Mutually exclusive with --poi.",
)
@click.option(
    "--on_target_box",
    type=click.Path(exists=True),
    default=None,
    help="Path to custom on-target box .txt file. Requires --on_target_receptor.",
)
@click.option("--sdf_file", type=click.Path(exists=True), default=None, help="SDF file with generated molecules")
@click.option("--smiles_file", type=click.Path(exists=True), default=None, help="Text file with one SMILES per line")
@click.option("--priority_targets", is_flag=True, help="Only consider priority safety targets")
@click.option("--targets", multiple=True,
              help="Only dock against these safety target case_names (repeatable, e.g. --targets kv111_inhibitor --targets nav15_antagonist). Omit for the full panel.")
@click.option("--nproc", default=_safe_nproc(), help="Number of processes for safety target docking")
@click.option("--exhaustiveness", default=EXHAUSTIVENESS_SAFETY, show_default=True,
              help="Vina exhaustiveness (paper protocol for the safety panel: 1, "
                   "chosen for throughput across the full anti-target set).")
@click.option("--fallback/--no-fallback", default=True, show_default=True,
              help="Disable the post-run fallback retry (nproc=1, lower "
                   "exhaustiveness, higher timeout) on failed (mol, target) "
                   "pair dockings. Fallback is ON by default.")
@click.option("--fallback_exhaustiveness", default=FALLBACK_EXHAUSTIVENESS_SAFETY,
              show_default=True, help="Vina exhaustiveness for fallback retries.")
@click.option("--fallback_timeout", default=FALLBACK_TIMEOUT_DEFAULT, show_default=True,
              help="Per-docking timeout (seconds) for fallback retries.")
@click.option("--output", type=click.Path(), default=None,
              help="Output CSV path. Default: results/evaluate-safety/<poi>.csv (or results/evaluate-safety/safety_eval.csv when using --on_target_receptor).")
def evaluate_safety(
    poi, on_target_receptor, on_target_box,
    sdf_file, smiles_file, priority_targets, targets, nproc, exhaustiveness,
    fallback, fallback_exhaustiveness, fallback_timeout,
    output,
):
    """
    Dock generated molecules against all safety targets (and optionally an on-target).

    For each molecule, reports the docking score against every safety target
    (``score_<target>``) plus, when an on-target is given, ``score_on_target``,
    a per-target delta ``delta_<target>`` (= score_on - score_<target>), the
    ``mean_safety_score`` across the panel, and ``delta_mean``
    (= score_on - mean_safety_score). Z-scores and aggregate metrics are
    computed at summary time (``scripts/summarize/cross_method.py``).

    On-target: pass exactly one of

    \b
      --poi <name>                              (protein of interest from standard_docking_benchmark.csv)
      --on_target_receptor + --on_target_box    (custom receptor files)

    Omit both to dock without an on-target score.

    \b
    Example:
      selectbench evaluate-safety \\
        --poi parp1 \\
        --sdf_file generated_molecules.sdf \\
        --output results/safety_eval.csv
    """
    if not sdf_file and not smiles_file:
        raise click.UsageError("Provide either --sdf_file or --smiles_file.")
    if sdf_file and smiles_file:
        raise click.UsageError("Provide only one of --sdf_file or --smiles_file.")

    # Mutex: --poi vs custom receptor/box pair
    _custom_on = bool(on_target_receptor or on_target_box)
    if poi and _custom_on:
        raise click.UsageError("Pass --poi OR --on_target_receptor/--on_target_box, not both.")
    if _custom_on and not (on_target_receptor and on_target_box):
        raise click.UsageError("--on_target_receptor and --on_target_box must be given together.")

    # Default output path
    if not output:
        if poi:
            output = os.path.join("results", "evaluate-safety", f"{poi}.csv")
        else:
            output = os.path.join("results", "evaluate-safety", "safety_eval.csv")

    # --- Resolve on-target receptor & box ---
    on_target_receptor_path = None
    on_target_box_path = None
    on_target_label = None

    if poi:
        sdb_cases = load_case_studies(os.path.join(SDB_DATA_DIR, "standard_docking_benchmark.csv"))
        if poi not in sdb_cases:
            raise click.BadParameter(
                f"Unknown POI: {poi}. Available: {', '.join(sdb_cases)}",
                param_hint="--poi",
            )
        case = sdb_cases[poi]
        pdb_id = case["pdb_id"]
        pdb_chain = case["pdb_chain_id"]
        processed_dir = os.path.join(SDB_DATA_DIR, "processed", poi)
        on_target_receptor_path = os.path.join(processed_dir, f"{pdb_id}_{pdb_chain}.pdbqt")
        on_target_box_path = os.path.join(processed_dir, f"{pdb_id}_{pdb_chain}.box.txt")
        on_target_label = f"{poi}/{pdb_id}_{pdb_chain}"
        if not os.path.exists(on_target_receptor_path) or not os.path.exists(on_target_box_path):
            raise FileNotFoundError(f"On-target receptor or box not found: {on_target_receptor_path}")
    elif on_target_receptor and on_target_box:
        on_target_receptor_path = on_target_receptor
        on_target_box_path = on_target_box
        on_target_label = os.path.basename(on_target_receptor).replace(".pdbqt", "")

    # --- Load safety targets ---
    st_cases = load_safety_cases(os.path.join(ST_DATA_DIR, "safety_targets.csv"))
    if priority_targets:
        st_cases = {k: v for k, v in st_cases.items() if int(v["priority_target"]) == 1}
    if targets:
        unknown = [t for t in targets if t not in st_cases]
        if unknown:
            raise click.UsageError(f"Unknown safety target(s): {', '.join(unknown)}. Available: {', '.join(st_cases)}")
        st_cases = {k: v for k, v in st_cases.items() if k in targets}

    safety_target_names = list(st_cases.keys())
    print(f"Safety targets: {len(safety_target_names)}")
    if on_target_label:
        print(f"On-target: {on_target_label}")

    # --- Load molecules ---
    if sdf_file:
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=True)
        smiles_list = [(i, Chem.MolToSmiles(mol)) for i, mol in enumerate(suppl) if mol is not None]
    else:
        with open(smiles_file) as f:
            smiles_list = [(i, line.strip()) for i, line in enumerate(f) if line.strip()]

    n_mols = len(smiles_list)
    print(f"Molecules to evaluate: {n_mols} | nproc={nproc}")

    # --- Pre-resolve safety target receptor/box paths ---
    st_receptors = {}
    for case_name, case_row in st_cases.items():
        pdb_id = case_row["pdb_id"]
        chain_id = case_row["pdb_chain_id"]
        if isinstance(chain_id, str) and chain_id.startswith("[") and chain_id.endswith("]"):
            chain_id = chain_id[1:-1].replace(";", "_")
        proc_dir = os.path.join(ST_DATA_DIR, "processed", case_name)
        receptor = os.path.join(proc_dir, f"{pdb_id}_{chain_id}.pdbqt")
        box_file = os.path.join(proc_dir, f"{pdb_id}_{chain_id}.box.txt")
        st_receptors[case_name] = (receptor, box_file)

    ctx = get_context("spawn")
    _ON_TARGET = "__on_target__"

    # --- Phase 1: Prepare all ligands in parallel ---
    with ctx.Pool(processes=nproc) as pool:
        prep_results = list(tqdm(
            pool.imap_unordered(_prep_ligand_worker, smiles_list),
            total=n_mols, desc="Preparing ligands", unit="mol",
        ))

    prep_map = {r["mol_idx"]: r for r in prep_results}
    for r in prep_results:
        if r["error"]:
            tqdm.write(f"  [{r['mol_idx']}] Ligand prep failed: {r['error']}")

    # --- Phase 2: Build flat docking task list ---
    docking_tasks = []
    for mol_idx, _ in smiles_list:
        r = prep_map[mol_idx]
        if r["error"]:
            continue
        pdbqt = r["pdbqt"]
        if on_target_receptor_path:
            docking_tasks.append((mol_idx, pdbqt, on_target_receptor_path, on_target_box_path, _ON_TARGET, exhaustiveness))
        for case_name, (receptor, box_file) in st_receptors.items():
            docking_tasks.append((mol_idx, pdbqt, receptor, box_file, case_name, exhaustiveness))

    # --- Phase 3: Dock all (molecule × target) tasks in one pool ---
    # Per-dock checkpointing: append each result to <output>.partial as it
    # finishes, then on resume skip tasks whose (mol_idx, target_label) key
    # is already in the partial. Survives walltime kills + node failures.
    dock_map = {mol_idx: {} for mol_idx, _ in smiles_list}
    partial_path = output + ".partial"
    sig_path = partial_path + ".sig"

    # A partial file is only reusable if the run configuration matches. The
    # per-dock key is (mol_idx, target_label), which says nothing about which
    # receptor or exhaustiveness produced a score — so resuming into the same
    # --output after changing --on_target_receptor or --exhaustiveness would
    # silently reuse the old protocol's numbers. Guard with a run signature.
    # The molecule set is part of the configuration: the per-dock key is
    # (mol_idx, target_label) and mol_idx is purely positional, so resuming a
    # partial written for a *different* input file would graft one molecule
    # set's scores onto another's SMILES without running a single docking.
    # Topping off a generated set and re-evaluating to the same --output is
    # exactly that case. The seed matters for the same reason: a seed sweep
    # resumed into an existing partial would silently mix seeds.
    # Hash index:SMILES pairs, not the SMILES alone: the partial is keyed on
    # mol_idx, so the same molecules in a different order are a different run.
    mol_digest = hashlib.sha256(
        "\n".join(f"{i}:{smi}" for i, smi in smiles_list).encode("utf-8")
    ).hexdigest()[:16]
    run_sig = "|".join([
        "on=" + (os.path.abspath(on_target_receptor_path)
                 if on_target_receptor_path else ""),
        f"exh={exhaustiveness}",
        "targets=" + ",".join(sorted(st_receptors)),
        f"mols={len(smiles_list)}:{mol_digest}",
        "seed=" + os.environ.get("SELECTBENCH_DOCK_SEED", "42"),
    ])
    if os.path.exists(partial_path):
        prev_sig = None
        if os.path.exists(sig_path):
            with open(sig_path) as sf:
                prev_sig = sf.read().strip()
        if prev_sig != run_sig:
            stale = f"{partial_path}.stale"
            os.replace(partial_path, stale)
            if os.path.exists(sig_path):
                os.remove(sig_path)
            print(f"  [resume] {partial_path} was written by a different run "
                  f"configuration (receptor / exhaustiveness / target set / "
                  f"molecule set / seed); moved it to {stale} and starting fresh.")

    done_keys = set()
    retry_keys = set()
    if os.path.exists(partial_path):
        with open(partial_path, newline="") as pf:
            rdr = csv.reader(pf)
            header = next(rdr, None)
            if header == ["mol_idx", "target_label", "score"]:
                for row in rdr:
                    if len(row) != 3:
                        continue
                    try:
                        m_idx = int(row[0])
                    except ValueError:
                        continue
                    label = row[1]
                    score = float(row[2]) if row[2] else None
                    if m_idx in dock_map:
                        dock_map[m_idx][label] = score
                        # Only a *successful* dock counts as done. Marking
                        # failures done retires them permanently, so a
                        # transient failure before a walltime kill would never
                        # be retried and would stay empty in the final CSV.
                        if score is None:
                            retry_keys.add((m_idx, label))
                        else:
                            done_keys.add((m_idx, label))
        if done_keys or retry_keys:
            msg = f"  [resume] {len(done_keys)} dockings recovered from {partial_path}"
            if retry_keys:
                msg += f"; retrying {len(retry_keys)} that had failed"
            print(msg)

    docking_tasks = [t for t in docking_tasks if (t[0], t[4]) not in done_keys]

    write_header = not os.path.exists(partial_path)
    os.makedirs(os.path.dirname(os.path.abspath(partial_path)) or ".", exist_ok=True)
    with open(sig_path, "w") as sf:
        sf.write(run_sig + "\n")
    with open(partial_path, "a", newline="") as pf:
        pwriter = csv.writer(pf)
        if write_header:
            pwriter.writerow(["mol_idx", "target_label", "score"])
            pf.flush()
        with NoDaemonPool(processes=nproc) as pool:
            for result in tqdm(
                pool.imap_unordered(_dock_pdbqt_worker, docking_tasks),
                total=len(docking_tasks), desc="Docking", unit="task",
            ):
                m_idx = result["mol_idx"]
                label = result["target_label"]
                score = result["score"]
                dock_map[m_idx][label] = score
                pwriter.writerow([m_idx, label, "" if score is None else score])
                pf.flush()
                if result["error"]:
                    tqdm.write(f"  [{m_idx}] [{label}] {result['error']}")

    # --- Phase 3.5: Fallback retry of failed (mol, target) dockings ---
    if fallback:
        receptor_box_by_target = dict(st_receptors)
        if on_target_receptor_path:
            receptor_box_by_target[_ON_TARGET] = (on_target_receptor_path, on_target_box_path)
        failed_before = {(m, lbl) for m, labels in dock_map.items()
                         for lbl, v in labels.items() if v is None}
        _retry_failed_safety_dockings(
            dock_map, prep_map, receptor_box_by_target,
            fallback_exhaustiveness, fallback_timeout,
        )
        # Checkpoint whatever the fallback recovered. Phase 3's partial-file
        # handle is already closed by this point, so without this the recovered
        # scores live only in memory and a kill during the (serial) fallback
        # loop loses all of them.
        recovered = [(m, lbl, dock_map[m][lbl]) for (m, lbl) in sorted(failed_before)
                     if dock_map[m].get(lbl) is not None]
        if recovered:
            with open(partial_path, "a", newline="") as pf:
                pwriter = csv.writer(pf)
                for m, lbl, score in recovered:
                    pwriter.writerow([m, lbl, score])
                pf.flush()
            print(f"  [fallback] checkpointed {len(recovered)} recovered docking(s)")

    # --- Phase 4: Assemble per-molecule rows ---
    rows = []
    for mol_idx, smiles in smiles_list:
        row = {"mol_idx": mol_idx, "smiles": smiles}
        prep_r = prep_map[mol_idx]

        if prep_r["error"]:
            if on_target_receptor_path:
                row["score_on_target"] = None
            for name in safety_target_names:
                row[f"score_{name}"] = None
            row["mean_safety_score"] = None
            if on_target_receptor_path:
                for name in safety_target_names:
                    row[f"delta_{name}"] = None
                row["delta_mean"] = None
                row["is_top20"] = None
            rows.append(row)
            continue

        mol_docks = dock_map[mol_idx]
        score_on = mol_docks.get(_ON_TARGET) if on_target_receptor_path else None
        if on_target_receptor_path:
            row["score_on_target"] = score_on

        safety_scores = []
        for name in safety_target_names:
            score = mol_docks.get(name)
            row[f"score_{name}"] = score
            if score is not None:
                safety_scores.append(score)
        row["mean_safety_score"] = sum(safety_scores) / len(safety_scores) if safety_scores else None
        # Panel coverage for this molecule. mean_safety_score averages only the
        # anti-targets that actually scored, and strict_selectivity_rate takes a
        # max over only those -- and a max over a subset is never larger than
        # over the full panel, so a molecule whose dockings mostly failed looks
        # *more* selective, not less. Recording the counts makes that visible
        # instead of silent; the usual cause is memory pressure killing the Vina
        # subprocess, which leaves no other trace in the CSV.
        row["n_safety_scored"] = len(safety_scores)
        row["n_safety_total"] = len(safety_target_names)

        if on_target_receptor_path:
            for name in safety_target_names:
                s = mol_docks.get(name)
                row[f"delta_{name}"] = (score_on - s) if score_on is not None and s is not None else None
            row["delta_mean"] = (
                score_on - row["mean_safety_score"]
                if score_on is not None and row["mean_safety_score"] is not None else None
            )
            # is_top20: is score_on in the top 20 % across on-target + all panel
            # scores for this molecule? Uses the single-source-of-truth
            # threshold helper so cli-emitted rows agree with the summary
            # recomputation in cross_method.py.
            if score_on is not None and safety_scores:
                from selectbench.metrics import top20_threshold_idx
                all_scores = sorted([score_on] + safety_scores)
                row["is_top20"] = int(score_on <= all_scores[top20_threshold_idx(len(all_scores))])
            else:
                row["is_top20"] = None

        on_str = f"  on={score_on:.2f}" if score_on is not None else ""
        mean_str = f"  mean_safety={row['mean_safety_score']:.2f}" if row["mean_safety_score"] is not None else ""
        delta_str = f"  delta_mean={row.get('delta_mean'):.2f}" if row.get("delta_mean") is not None else ""
        print(f"  [{mol_idx}]{on_str}{mean_str}{delta_str}")

        rows.append(row)

    # Raw per-mol CSV only — aggregate metrics are computed on demand by
    # scripts/summarize/. Print a one-line console summary for convenience.
    scored = [r for r in rows if r.get("mean_safety_score") is not None]
    if scored and on_target_receptor_path:
        from selectbench.metrics import safety_metrics
        m = safety_metrics(rows, safety_target_names)
        on = m.get('score_on_target')
        msf = m.get('mean_safety_score')
        dm = m.get('delta_mean')
        sr_mean = m.get('selectivity_rate_1kcal_mean')
        strict = m.get('strict_selectivity_rate_1kcal')
        def _f(v): return f"{v:.2f}" if v is not None else "—"
        print(f"\n  safety n={m['n_valid']}/{m['n_total']}  "
              f"on={_f(on)}  mean_safety={_f(msf)}  Δ_mean={_f(dm)}  "
              f"SR_1k_mean={_f(sr_mean)}  strict_SR_1k={_f(strict)}")

    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fieldnames = list(dict.fromkeys(k for row in rows for k in row.keys()))
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {output}")
        # Final CSV is durable now — drop the per-dock checkpoint and its
        # run-signature sidecar together, so a later run doesn't find an
        # orphaned .sig with no .partial beside it.
        for path in (partial_path, sig_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# compute-reference-stats: dock reference ligands to build Z-score statistics
# ---------------------------------------------------------------------------

def _dock_reference_ligands(receptor_path, box_file_path, smiles_list):
    """Dock a list of SMILES against a single receptor, return list of scores.

    Uses ``run_vina_safe`` (subprocess-isolated with a timeout) so one hanging
    reference ligand can't stall the whole reference-stats build.
    """
    scores = []
    for i, smiles in enumerate(smiles_list):
        try:
            ligand_pdbqt = prepare_ligand(smiles)
            out = run_vina_safe(ligand_pdbqt, receptor_path, box_file_path)
            scores.append(float(out[0][0]))
        except Exception as e:
            print(f"    [ref ligand {i}] failed: {e}")
    return scores


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


@cli.command("compute-reference-stats")
@click.option("--case_study", default=None,
              help="Compute stats for both receptors of this case study.")
@click.option("--safety_targets", "use_safety_targets", is_flag=True,
              help="Compute stats for all safety targets.")
@click.option("--priority_targets", is_flag=True,
              help="Only consider priority safety targets (used with --safety_targets).")
@click.option("--targets", multiple=True, default=None,
              help="Restrict to these safety target case_names (used with --safety_targets).")
@click.option("--receptor", "custom_receptor", type=click.Path(exists=True), default=None,
              help="Path to a custom receptor .pdbqt file.")
@click.option("--box", "custom_box", type=click.Path(exists=True), default=None,
              help="Path to the matching docking box .txt file.")
@click.option("--target_id", default=None,
              help="target_id label to use when writing stats for a custom receptor.")
@click.option("--sdf_file", type=click.Path(exists=True), default=None,
              help="SDF file of reference ligands.")
@click.option("--smiles_file", type=click.Path(exists=True), default=None,
              help="Text file with one reference SMILES per line.")
@click.option("--output", required=True, type=click.Path(),
              help="Output CSV path (appended to if it already exists).")
def compute_reference_stats(
    case_study, use_safety_targets, priority_targets, targets,
    custom_receptor, custom_box, target_id,
    sdf_file, smiles_file, output,
):
    """
    Dock a set of reference ligands against one or more targets and write the
    per-target mean and std of docking scores to a CSV. The output file feeds
    the summary tooling (``scripts/summarize/cross_method.py``) via its
    ``--cs_reference_stats`` / ``--safety_reference_stats`` /
    ``--crossdocked_reference_stats`` flags to compute Z-score columns.

    \b
    Targets can be specified as:
      --case_study <name>        both receptors of a case study
      --safety_targets           all (or filtered) safety targets
      --receptor + --box         a single custom receptor

    These can be combined in one call to build a combined stats file.

    \b
    Example:
      selectbench compute-reference-stats \\
        --case_study parp \\
        --safety_targets \\
        --sdf_file reference_ligands.sdf \\
        --output stats/reference_stats.csv
    """
    if not sdf_file and not smiles_file:
        raise click.UsageError("Provide either --sdf_file or --smiles_file.")
    if sdf_file and smiles_file:
        raise click.UsageError("Provide only one of --sdf_file or --smiles_file.")
    if not case_study and not use_safety_targets and not custom_receptor:
        raise click.UsageError(
            "Specify at least one target via --case_study, --safety_targets, or --receptor/--box."
        )
    if custom_receptor and not custom_box:
        raise click.UsageError("--receptor requires --box.")
    if custom_receptor and not target_id:
        raise click.UsageError("--receptor requires --target_id.")

    smiles_list = _load_smiles_list(sdf_file, smiles_file)
    print(f"Reference ligands loaded: {len(smiles_list)}")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    # --- Case study receptors ---
    if case_study:
        cs_cases = load_case_studies(os.path.join(CS_DATA_DIR, "case_studies.csv"))
        if case_study not in cs_cases:
            raise click.BadParameter(f"Unknown case study: {case_study}", param_hint="--case_study")
        case = cs_cases[case_study]
        processed_dir = os.path.join(CS_DATA_DIR, case_study, "processed")
        for pdb_key in ("pdb1", "pdb2"):
            pdb_id = case[f"{pdb_key}_id"]
            chain_id = case[f"{pdb_key}_chain_id"]
            tid = f"{pdb_id}_{chain_id}"
            receptor_path = os.path.join(processed_dir, f"{tid}.pdbqt")
            box_path = os.path.join(processed_dir, f"{tid}.box.txt")
            if not os.path.exists(receptor_path) or not os.path.exists(box_path):
                print(f"  [skip] {tid}: receptor or box not found")
                continue
            print(f"  Docking {len(smiles_list)} reference ligands against {tid} ...")
            scores = _dock_reference_ligands(receptor_path, box_path, smiles_list)
            mean, std = _stats_from_scores(scores)
            if mean is None:
                print(f"  [skip] {tid}: fewer than 2 valid scores")
                continue
            _append_stats_row(output, tid, mean, std)
            print(f"  {tid}: mean={mean:.3f}  std={std:.3f}  (n={len(scores)})")

    # --- Safety targets ---
    if use_safety_targets:
        st_cases = load_safety_cases(os.path.join(ST_DATA_DIR, "safety_targets.csv"))
        if priority_targets:
            st_cases = {k: v for k, v in st_cases.items() if int(v["priority_target"]) == 1}
        if targets:
            unknown = [t for t in targets if t not in st_cases]
            if unknown:
                raise click.UsageError(
                    f"Unknown safety target(s): {', '.join(unknown)}. "
                    f"Available: {', '.join(st_cases)}"
                )
            st_cases = {k: v for k, v in st_cases.items() if k in targets}

        for case_name, case_row in st_cases.items():
            pdb_id = case_row["pdb_id"]
            chain_id = case_row["pdb_chain_id"]
            if isinstance(chain_id, str) and chain_id.startswith("[") and chain_id.endswith("]"):
                chain_id = chain_id[1:-1].replace(";", "_")
            processed_dir = os.path.join(ST_DATA_DIR, "processed", case_name)
            receptor_path = os.path.join(processed_dir, f"{pdb_id}_{chain_id}.pdbqt")
            box_path = os.path.join(processed_dir, f"{pdb_id}_{chain_id}.box.txt")
            if not os.path.exists(receptor_path) or not os.path.exists(box_path):
                print(f"  [skip] {case_name}: receptor or box not found")
                continue
            print(f"  Docking {len(smiles_list)} reference ligands against {case_name} ...")
            scores = _dock_reference_ligands(receptor_path, box_path, smiles_list)
            mean, std = _stats_from_scores(scores)
            if mean is None:
                print(f"  [skip] {case_name}: fewer than 2 valid scores")
                continue
            _append_stats_row(output, case_name, mean, std)
            print(f"  {case_name}: mean={mean:.3f}  std={std:.3f}  (n={len(scores)})")

    # --- Custom receptor ---
    if custom_receptor:
        print(f"  Docking {len(smiles_list)} reference ligands against {target_id} ...")
        scores = _dock_reference_ligands(custom_receptor, custom_box, smiles_list)
        mean, std = _stats_from_scores(scores)
        if mean is None:
            print(f"  [skip] {target_id}: fewer than 2 valid scores")
        else:
            _append_stats_row(output, target_id, mean, std)
            print(f"  {target_id}: mean={mean:.3f}  std={std:.3f}  (n={len(scores)})")

    print(f"\nReference stats written to {output}")


@cli.command()
@click.option("--case_study", required=True, help="Name of the case study")
@click.option("--smiles", required=True, help="Input ligand SMILES")
@click.option(
    "--output_dir",
    type=click.Path(),
    default=None,
    help="Directory to save Boltz2 prediction outputs",
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Use the MSA server for MSA generation",
)
def check_boltz2(case_study, smiles, output_dir, use_msa_server):
    """
    Run Boltz2 affinity prediction for a SMILES ligand on the two receptors of a case study.
    """
    # boltz2 pulls in heavy deps (boltz, yaml); import lazily so non-boltz2
    # commands work even when those aren't installed.
    from selectbench.data.boltz2_affinity import run_boltz2

    csv_path = os.path.join(CS_DATA_DIR, "case_studies.csv")
    cases = load_case_studies(csv_path)

    if case_study not in cases:
        raise click.BadParameter(f"Unknown case study: {case_study}", param_hint="--case_study")

    case = cases[case_study]
    processed_dir = os.path.join(CS_DATA_DIR, case_study, "processed")

    pdb1_id = case["pdb1_id"]
    pdb1_chain = case["pdb1_chain_id"]
    pdb2_id = case["pdb2_id"]
    pdb2_chain = case["pdb2_chain_id"]

    pdb1_path = os.path.join(processed_dir, f"{pdb1_id}_{pdb1_chain}.pdb")
    pdb2_path = os.path.join(processed_dir, f"{pdb2_id}_{pdb2_chain}.pdb")

    if not all(os.path.exists(p) for p in [pdb1_path, pdb2_path]):
        raise FileNotFoundError("Processed receptor PDB not found")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    out1_dir = os.path.join(output_dir, f"{pdb1_id}_{pdb1_chain}") if output_dir else None
    out2_dir = os.path.join(output_dir, f"{pdb2_id}_{pdb2_chain}") if output_dir else None

    result1 = run_boltz2(smiles, pdb1_path, chain_id=pdb1_chain, output_dir=out1_dir, use_msa_server=use_msa_server)
    result2 = run_boltz2(smiles, pdb2_path, chain_id=pdb2_chain, output_dir=out2_dir, use_msa_server=use_msa_server)

    print(f"{pdb1_id}_{pdb1_chain}: {result1}")
    print(f"{pdb2_id}_{pdb2_chain}: {result2}")
    print("Boltz2 prediction completed.")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), default=None,
              help="Per-molecule CSV to write "
                   "(mol_idx,smiles,qed,sa,logp,lipinski,mw). "
                   "Default: results/quality/<input-stem>.csv.")
def quality(input_file, output):
    """Compute reference-free molecular-quality metrics for a SMILES/SDF file.

    QED, normalized SA, LogP, Lipinski count, MW (per molecule) plus set-level
    Diversity — no docking required. INPUT_FILE may be a .sdf, .smi/.txt (one
    SMILES per line), or .csv (with a 'smiles' column). Writes the per-molecule
    table to --output and prints the aggregate (means/stds + diversity).
    """
    from selectbench.sbdd_metrics import (
        load_smiles_from_file, compute_per_mol_metrics, compute_aggregate_metrics,
    )

    smiles = load_smiles_from_file(input_file)
    if not smiles:
        raise click.ClickException(f"No SMILES found in {input_file}")

    per_mol = compute_per_mol_metrics(smiles)
    agg = compute_aggregate_metrics(per_mol)

    if output is None:
        stem = os.path.splitext(os.path.basename(input_file))[0]
        output = os.path.join("results", "quality", f"{stem}.csv")
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fieldnames = ["mol_idx", "smiles", "qed", "sa", "logp", "lipinski", "mw"]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_mol)

    print(f"Wrote {output}  ({agg['n_scored']}/{agg['n_total']} molecules scored)")
    print("Aggregate quality metrics:")
    for k in ("qed_mean", "qed_std", "sa_mean", "sa_std", "logp_mean", "logp_std",
              "lipinski_mean", "lipinski_std", "mw_mean", "mw_std", "diversity"):
        print(f"  {k:16s} {agg[k]}")


if __name__ == "__main__":
    cli()
