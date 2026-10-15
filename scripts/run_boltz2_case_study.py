"""
Run Boltz2 affinity prediction for all ligands in a file against the two
receptors of a selectbench case study — the batch, case-study-scale
counterpart to the single-SMILES ``selectbench check-boltz2`` CLI command.

Pocket constraints are auto-discovered from *_pocket.pdb files.
Results are written row-by-row so progress is preserved on crash.
Re-running resumes from where it left off.

--input takes your generator's output directly — no prior Vina eval required.
Three formats are accepted, dispatched by extension:
  * ``.sdf``      — a multi-molecule SDF (RDKit-parseable);
  * ``.smi``/``.txt`` — one SMILES per line (blank lines skipped);
  * ``.csv``      — must have a ``smiles`` column (``mol_idx`` used if present).
SDF and SMILES-list handling mirror ``selectbench evaluate-case``'s
``_dock_sdf`` / ``_dock_smiles_file``, so ``mol_idx`` lines up 1-to-1 with the
Vina eval on the same file (handy if you later want to compare the two scorers).

The output goes to ``results/boltz2/<model>/cs/<case>.csv`` and the
cross-method summarizer keys each row's *model* off that ``<model>``
**directory name** — NOT the input filename. To label results by method, pass
``--model <method>`` (e.g. ``--model diffsbdd``); it sets the ``<model>`` dir for
you. Falling back to the default (``<model>`` = the input filename stem) labels
correctly only when the input is named ``<method>.<ext>`` — feeding
``generated_samples/diffsbdd/cs/parp.sdf`` without ``--model`` would mislabel the
model as ``parp`` (the case name). ``--output`` overrides both if you need a
fully custom path.

Usage:
    # Feed a generated SDF straight from your model; label the run with --model
    python scripts/run_boltz2_case_study.py \
        --case_study cdk2_cdk9_jcim \
        --input generated_samples/diffsbdd/cs/cdk2_cdk9_jcim.sdf \
        --model diffsbdd \
        [--output_dir results/boltz2_outputs/cdk2_cdk9_jcim] \
        [--nproc 4] \
        [--use_msa_server]

    # Then summarize across methods:
    #   python scripts/summarize/cross_method.py --score_fn boltz2

Env vars:
    BOLTZ2_MSA_CACHE_DIR   directory for cached per-protein MSAs
                           (default: results/boltz2_msa_cache)
    BOLTZ2_OUTPUT_DIR      per-run raw Boltz2 outputs
                           (default: results/boltz2_outputs/<case_study>)
"""

import argparse
import csv
import os
import shutil
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_STUDIES_CSV = os.path.join(REPO_ROOT, "data", "case_studies", "case_studies.csv")
PROCESSED_BASE   = os.path.join(REPO_ROOT, "data", "case_studies")

FIELDNAMES = [
    "mol_idx", "smiles",
    "on_target_affinity_prob",  "on_target_affinity_value",  "on_target_error",
    "off_target_affinity_prob", "off_target_affinity_value", "off_target_error",
]


def load_case(case_name):
    df = pd.read_csv(CASE_STUDIES_CSV)
    row = df[df["case_name"] == case_name]
    if row.empty:
        raise ValueError(f"Unknown case study '{case_name}'. Available: {df['case_name'].tolist()}")
    row = row.iloc[0]
    processed_dir = os.path.join(PROCESSED_BASE, case_name, "processed")
    on_target = row["on_target"]  # "pdb1" or "pdb2"
    targets = {}
    for label, key in [("on_target", on_target), ("off_target", "pdb2" if on_target == "pdb1" else "pdb1")]:
        pdb_id    = row[f"{key}_id"]
        chain_id  = row[f"{key}_chain_id"]
        pdb_path  = os.path.join(processed_dir, f"{pdb_id}_{chain_id}.pdb")
        targets[label] = {"pdb_path": pdb_path, "chain_id": chain_id, "pdb_id": pdb_id}
    return targets


def load_ligands(input_path):
    """Load ligands from a generated SDF, a SMILES list (.smi/.txt), or a CSV
    with a ``smiles`` column → DataFrame with ``mol_idx`` + ``smiles``.

    No prior docking eval is required — point this straight at your generator's
    output. The SDF and SMILES-list handling mirror ``selectbench
    evaluate-case``'s ``_dock_sdf`` / ``_dock_smiles_file`` (RDKit canonical
    SMILES for SDF, one SMILES per non-blank line for text), so ``mol_idx``
    lines up with the Vina eval on the same input file.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".sdf":
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(input_path)
        rows = [(i, Chem.MolToSmiles(mol))
                for i, mol in enumerate(suppl) if mol is not None]
        return pd.DataFrame(rows, columns=["mol_idx", "smiles"])
    if ext in (".smi", ".txt"):
        with open(input_path) as f:
            rows = [(i, line.strip()) for i, line in enumerate(f) if line.strip()]
        return pd.DataFrame(rows, columns=["mol_idx", "smiles"])
    if ext == ".csv":
        df = pd.read_csv(input_path)
        if "smiles" not in df.columns:
            raise ValueError(f"CSV must have a 'smiles' column; found: {list(df.columns)}")
        if "mol_idx" not in df.columns:
            df["mol_idx"] = df.index
        return df
    raise ValueError(f"Unsupported --input extension '{ext}'. Use .sdf, .smi/.txt, or .csv.")


def _run_one(mol_idx, smiles, targets, output_dir, use_msa_server, msa_cache_dir):
    import sys
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    from selectbench.data.boltz2_affinity import run_boltz2

    row = {"mol_idx": mol_idx, "smiles": smiles}

    for label, info in targets.items():
        target_out = None
        if output_dir:
            target_out = os.path.join(output_dir, f"mol{mol_idx}_{label}")
            os.makedirs(target_out, exist_ok=True)
        try:
            result = run_boltz2(
                smiles,
                info["pdb_path"],
                chain_id=info["chain_id"],
                output_dir=target_out,
                use_msa_server=use_msa_server,
                pocket_pdb_path="auto",
                msa_cache_dir=msa_cache_dir,
            )
            row[f"{label}_affinity_prob"]  = result["affinity_probability_binary"]
            row[f"{label}_affinity_value"] = result["affinity_pred_value"]
        except Exception as e:
            row[f"{label}_affinity_prob"]  = None
            row[f"{label}_affinity_value"] = None
            row[f"{label}_error"]          = str(e)
            print(f"[mol {mol_idx} {label}] ERROR: {e}")

    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--case_study", required=True, help="Case study name (e.g. cdk2_cdk9_jcim)")
    parser.add_argument("--input",      required=True,
                        help="Ligands to score: a generated .sdf, a .smi/.txt SMILES list "
                             "(one per line), or a .csv with a 'smiles' column. Feed your "
                             "generator's output directly — no prior Vina eval required.")
    parser.add_argument("--model",      default=None,
                        help="Method label for this run (e.g. diffsbdd). Sets the "
                             "<model> dir the summarizer keys off — output becomes "
                             "results/boltz2/<model>/cs/<case_study>.csv. Recommended "
                             "over relying on the --input filename stem. Ignored if "
                             "--output is given.")
    parser.add_argument("--output",     default=None,
                        help="Explicit path to write results CSV. Overrides --model. "
                             "Default: results/boltz2/<model>/cs/<case_study>.csv, where "
                             "<model> comes from --model, else the --input filename stem. "
                             "The summarizer keys the model off the <model> dir name.")
    parser.add_argument("--output_dir", default=None,  help="Directory for raw Boltz2 outputs")
    parser.add_argument("--nproc",      type=int, default=1)
    parser.add_argument("--use_msa_server", action="store_true")
    parser.add_argument("--msa_cache_dir",
                        default=os.environ.get("BOLTZ2_MSA_CACHE_DIR",
                                               os.path.join(REPO_ROOT, "results", "boltz2_msa_cache")),
                        help="Directory to cache per-protein MSAs. If an MSA exists here, the "
                             "server is skipped; otherwise --use_msa_server is used and the result "
                             "is cached for future runs. (env: BOLTZ2_MSA_CACHE_DIR)")
    args = parser.parse_args()

    if shutil.which("boltz") is None:
        sys.exit("ERROR: `boltz` not found on PATH. Install with "
                 "`pip install boltz` (see BOLTZ2.md).")

    model_name = args.model or os.path.splitext(os.path.basename(args.input))[0]
    if args.output is None:
        args.output = os.path.join(REPO_ROOT, "results", "boltz2", model_name, "cs", f"{args.case_study}.csv")
    output_parent = os.path.dirname(args.output)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    if args.output_dir is None:
        # Raw intermediates keyed by both case and model so two models scored on
        # the same case don't collide in the same mol<idx>_<label> subdirs.
        args.output_dir = os.environ.get(
            "BOLTZ2_OUTPUT_DIR",
            os.path.join(REPO_ROOT, "results", "boltz2_outputs", args.case_study, model_name),
        )

    targets = load_case(args.case_study)

    df = load_ligands(args.input)

    os.makedirs(args.output_dir, exist_ok=True)

    # Resume: skip mol_idx values where both targets completed without error
    done = set()
    error_cols = ["on_target_error", "off_target_error"]
    if os.path.exists(args.output):
        existing = pd.read_csv(args.output)
        # Older partial runs may be missing error columns entirely — treat those
        # runs as all-successful (nothing to filter on).
        for col in error_cols:
            if col not in existing.columns:
                existing[col] = float("nan")
        successful = existing[existing[error_cols].isna().all(axis=1)]
        done = set(successful["mol_idx"].tolist())
        n_failed = len(existing) - len(successful)
        print(f"Resuming — {len(done)} molecules already done, {n_failed} failed (will retry), skipping.")

    pending = df[~df["mol_idx"].isin(done)]
    # Capture now: `done` is mutated as rows are written below, so computing
    # the total afterwards double-counts everything that was resumed.
    n_total_expected = len(done) + len(pending)

    # Rewrite the file with only successful rows to drop stale error rows before appending
    if os.path.exists(args.output):
        # reindex to FIELDNAMES first: error columns synthesised above land at
        # the end of the frame, and the append below writes in FIELDNAMES order,
        # so without this the appended rows sit under the wrong headers.
        (successful.drop_duplicates(subset="mol_idx", keep="last")
                   .sort_values("mol_idx")
                   .reindex(columns=FIELDNAMES)
                   .to_csv(args.output, index=False))
        write_header = False
    else:
        write_header = True

    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        if args.nproc > 1:
            import multiprocessing
            tasks = [
                (row["mol_idx"], row["smiles"], targets, args.output_dir, args.use_msa_server, args.msa_cache_dir)
                for _, row in pending.iterrows()
            ]
            with multiprocessing.Pool(args.nproc) as pool:
                for result_row in pool.starmap(_run_one, tasks):
                    if int(result_row["mol_idx"]) not in done:
                        writer.writerow(result_row)
                        done.add(int(result_row["mol_idx"]))
                        f.flush()
        else:
            for _, row in pending.iterrows():
                if int(row["mol_idx"]) in done:
                    continue
                result_row = _run_one(row["mol_idx"], row["smiles"], targets, args.output_dir, args.use_msa_server, args.msa_cache_dir)
                if int(result_row["mol_idx"]) not in done:
                    writer.writerow(result_row)
                    done.add(int(result_row["mol_idx"]))
                    f.flush()

    print(f"Done. {n_total_expected} molecules total in {args.output}")


if __name__ == "__main__":
    main()
