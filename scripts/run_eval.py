#!/usr/bin/env python3
"""
Evaluate generated molecules for a given method.

Input layout (auto-discovered):
    generated_samples/<method>/cs/<case>.{sdf,smi}                       (--eval case)
    generated_samples/<method>/safety/<poi>.{sdf,smi}                    (--eval safety)
    generated_samples/<method>/crossdocked/<stem>.{sdf,smi}              (--eval crossdocked)

Output layout (parallel):
    results/<method>/cs/<case>.csv
    results/<method>/safety/<poi>.csv
    results/<method>/crossdocked/<on>_vs_<off>__<diff>[_<sweep>].csv

Crossdocked stems supported:
    <on>                                # bare on-target pocket name (e.g.
                                        #   3tym_A); fans out to every off-pocket
                                        #   that pairs with <on>, across all tiers.
                                        #   This is what the gen_* wrappers emit.
    <on>__<diff>                        # baseline; fans out to every off-pocket
                                        #   that pairs with <on> at <diff> only
    <on>_vs_<off>__<diff>[_<sweep>]     # compose; one case_id per stem
                                        #   (sweep tag passes through to CSV name)

Usage:
    # Case study eval — all SDFs under generated_samples/diffsbdd/cs/
    python scripts/run_eval.py --method diffsbdd --eval case

    # Safety eval — POI inferred from filename or --poi
    python scripts/run_eval.py --method diffsbdd --eval safety --poi parp1

    # Crossdocked eval — auto-discovers generated_samples/<method>/crossdocked/
    python scripts/run_eval.py --method diffsbdd --eval crossdocked

    # Single file
    python scripts/run_eval.py --method diffsbdd --eval case \\
        --sdf generated_samples/diffsbdd/cs/parp.sdf

    # Dry run
    python scripts/run_eval.py --method diffsbdd --eval case --dry_run
"""

import os
import re
import sys
import glob
import argparse
import subprocess
import csv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASE_STUDIES_CSV = os.path.join(REPO_ROOT, "data", "case_studies", "case_studies.csv")
SDB_CSV = os.path.join(REPO_ROOT, "data", "standard_docking_benchmark", "standard_docking_benchmark.csv")
CROSSDOCKED_CSV = os.path.join(REPO_ROOT, "data", "crossdocked_test_set",
                               "crossdocked_test_set_pairs.csv")


def load_case_study_names():
    with open(CASE_STUDIES_CSV) as f:
        return [r["case_name"] for r in csv.DictReader(f)]


def load_poi_names():
    with open(SDB_CSV) as f:
        return [r["case_name"] for r in csv.DictReader(f)]


def infer_case_study(sdf_path, known_case_studies):
    """Infer case study name from a filename stem (exact, else longest prefix).

    A prefix must end on a ``_`` boundary: ``parp_seed1`` resolves to ``parp``,
    but ``parp1`` and ``parpXYZ`` resolve to nothing. Unanchored matching would
    silently route the PoI ``parp1`` to the case study ``parp`` -- a different
    protein and a different receptor pair -- and the run would dock for hours
    and emit a confidently mislabelled CSV. Names that resolve to nothing are
    reported by the caller and make the run exit non-zero.
    """
    stem = os.path.splitext(os.path.basename(sdf_path))[0]
    if stem in known_case_studies:
        return stem
    for name in sorted(known_case_studies, key=len, reverse=True):
        if stem.startswith(name + "_"):
            return name
    return None


def load_crossdocked_indices():
    """Return (case_info, on_index, pair_index).

    - case_info[case_id]   = (on_name, off_name, difficulty)
    - on_index[on_name]    = [(case_id, off_name, difficulty), ...]
    - pair_index[(on, off, diff)] = case_id
    """
    case_info = {}
    on_index = {}
    pair_index = {}
    with open(CROSSDOCKED_CSV) as f:
        for r in csv.DictReader(f):
            cid = int(r["case_id"])
            diff = r["difficulty"]
            on_k = "pdb1" if r["on_target"] == "pdb1" else "pdb2"
            off_k = "pdb2" if r["on_target"] == "pdb1" else "pdb1"
            on = f"{r[f'{on_k}_id']}_{r[f'{on_k}_chain_id']}"
            off = f"{r[f'{off_k}_id']}_{r[f'{off_k}_chain_id']}"
            case_info[cid] = (on, off, diff)
            on_index.setdefault(on, []).append((cid, off, diff))
            pair_index[(on, off, diff)] = cid
    return case_info, on_index, pair_index


# Compose stem: <on>_vs_<off>__<diff>[_<sweep>]
# on/off look like "<pdb>_<chain>" (e.g. 2rhy_A) — the chain is a single
# alphanum after a single underscore, anchored by the trailing "_vs_".
_COMPOSE_STEM_RE = re.compile(
    r"^(?P<on>[^_]+_[^_]+)_vs_(?P<off>[^_]+_[^_]+)__(?P<diff>easy|medium|hard)"
    r"(?:_(?P<sweep>.+))?$"
)
# Baseline stem: <on>__<diff>
_BASELINE_STEM_RE = re.compile(
    r"^(?P<on>[^_]+_[^_]+)__(?P<diff>easy|medium|hard)$"
)


def crossdocked_jobs_for(stem, on_index, pair_index):
    """Return [(case_id, csv_basename), ...] for one input stem.

    Compose stems map 1-to-1 to a case_id (with sweep tag preserved in the
    CSV name). Baseline stems fan out across every off-pocket that pairs
    with <on> at the same difficulty.
    """
    m = _COMPOSE_STEM_RE.match(stem)
    if m:
        on, off, diff, sweep = m["on"], m["off"], m["diff"], m["sweep"]
        cid = pair_index.get((on, off, diff))
        if cid is None:
            return []
        suffix = f"_{sweep}" if sweep else ""
        return [(cid, f"{on}_vs_{off}__{diff}{suffix}.csv")]

    m = _BASELINE_STEM_RE.match(stem)
    if m:
        on, diff = m["on"], m["diff"]
        return [(cid, f"{on}_vs_{off}__{diff}.csv")
                for cid, off, d in on_index.get(on, []) if d == diff]

    # Bare on-target stem (no difficulty) — fan out across every pair.
    if stem in on_index:
        return [(cid, f"{stem}_vs_{off}__{d}.csv")
                for cid, off, d in on_index[stem]]
    return []


def run_cmd(cmd, dry_run=False, failures=None):
    print(f"  $ {' '.join(cmd)}")
    if not dry_run:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  [WARNING] Command exited with code {result.returncode}")
            if failures is not None:
                failures.append((result.returncode, cmd))


def _discover(method, kind):
    """List input files in generated_samples/<method>/<kind>/."""
    base = os.path.join(REPO_ROOT, "generated_samples", method, kind)
    return sorted(glob.glob(os.path.join(base, "*.sdf"))
                  + glob.glob(os.path.join(base, "*.smi")))


def auto_discover_methods(kind):
    """Return every generated_samples/<name>/ subdir holding at least one
    SDF or SMI under <name>/<kind>/. Used when --method isn't passed."""
    root = os.path.join(REPO_ROOT, "generated_samples")
    if not os.path.isdir(root):
        return []
    found = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if _discover(name, kind):
            found.append(name)
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated molecules for a method.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--method", "--methods", dest="method", nargs="+", default=None,
                        help="Method name(s) (e.g. diffsbdd, diffsbdd_compose). "
                             "Default: auto-discover every generated_samples/<name>/ "
                             "that has files in the requested <kind>/ subdir. "
                             "(Alias --methods accepted for cross_method.py compat.)")
    parser.add_argument("--sdf", "--sdf_file", "--input", dest="sdf",
                        nargs="+", default=None,
                        help="Path(s) to SDF/SMI file(s). Repeatable. Skips auto-discovery. "
                             "(Aliases --sdf_file and --input accepted.)")
    parser.add_argument("--case_study", default=None,
                        help="Override case study name (inferred from filename if omitted).")
    parser.add_argument("--poi", nargs="+", default=None,
                        help="Protein(s) of interest for safety eval (e.g. parp1 jak2).")
    parser.add_argument("--eval", choices=["cs", "case", "safety", "crossdocked"], default="cs",
                        help="Which benchmark to run (default: cs). 'case' is a "
                             "back-compat alias for 'cs' (the summarizer and bucket "
                             "dirs both use 'cs').")
    parser.add_argument("--priority_targets", action="store_true",
                        help="(--eval safety only) Restrict to the priority safety subset.")
    parser.add_argument("--targets", action="append", default=None, metavar="TARGET",
                        help="(--eval safety only) Specific safety target(s); repeatable.")
    parser.add_argument("--nproc", type=int, default=None,
                        help="Docking workers. Default: min(8, SLURM_CPUS_PER_TASK or cpu_count()) "
                             "— safer than cpu_count() on shared clusters. Pass 0 to explicitly hand "
                             "off to the underlying CLI's default (grabs every core).")
    parser.add_argument("--exhaustiveness", type=int, default=None,
                        help="Vina exhaustiveness. Per-eval defaults: "
                             "case=20, crossdocked=10, safety=1.")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-molecule docking timeout in seconds "
                             "(default: 600). Exported as "
                             "SELECTBENCH_DOCK_TIMEOUT to the docking subprocess.")
    parser.add_argument("--out_csv", default=None,
                        help="Override output CSV filename (joined with the eval-type subdir).")
    parser.add_argument("--case_ids", default=None,
                        help="(crossdocked only) Comma-separated case_ids to override "
                             "the auto-fan-out from the SDF stem.")
    parser.add_argument("--no_fallback", "--no-fallback", dest="no_fallback",
                        action="store_true",
                        help="Disable the low-exhaustiveness retry for failed dockings "
                             "(forwarded as --no-fallback to each selectbench subcommand).")
    parser.add_argument("--results_name", default=None,
                        help="Custom results subdirectory name (default: same as --method)")
    parser.add_argument("--results_dir", default=None,
                        help="Base directory for results (default: <repo>/results). "
                             "Output lands in <results_dir>/<method>/<bucket>/.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the resolved dispatch plan without running anything.")
    args = parser.parse_args()

    # 'case' is a back-compat alias; normalize to the canonical 'cs' used by the
    # bucket dirs and cross_method.py --mode.
    if args.eval == "case":
        args.eval = "cs"

    # Reject --priority_targets / --targets outside of safety mode (silently no-op'd before).
    if args.eval != "safety" and (args.priority_targets or args.targets):
        parser.error("--priority_targets and --targets only apply to --eval safety.")

    # --sdf existence check: without this, typos surface as a stack trace deep
    # in the subprocess pipeline instead of a friendly argparse error.
    if args.sdf:
        missing = [p for p in args.sdf if not os.path.isfile(p)]
        if missing:
            parser.error(f"--sdf file(s) not found: {', '.join(missing)}")

    # --case_ids: validate up front (numeric, crossdocked-only, non-empty) so a
    # typo surfaces as a clean parser error rather than a ValueError deep inside
    # the fan-out loop. Range is checked later against the pairs CSV.
    args.case_ids_parsed = None
    if args.case_ids is not None:
        if args.eval != "crossdocked":
            parser.error("--case_ids only applies to --eval crossdocked.")
        try:
            args.case_ids_parsed = [int(x) for x in args.case_ids.split(",") if x.strip()]
        except ValueError:
            parser.error(f"--case_ids must be comma-separated integers, got: {args.case_ids!r}")
        if not args.case_ids_parsed:
            parser.error("--case_ids parsed to an empty list.")

    known_case_studies = load_case_study_names()
    known_pois = load_poi_names()

    kind = {"cs": "cs", "safety": "safety", "crossdocked": "crossdocked"}[args.eval]

    # Per-eval-mode exhaustiveness defaults. CLI --exhaustiveness wins.
    if args.exhaustiveness is None:
        args.exhaustiveness = {"cs": 20, "crossdocked": 10, "safety": 1}[args.eval]

    # nproc default: prefer SLURM_CPUS_PER_TASK, cap at 8 to protect shared clusters
    # from silently grabbing every core (the underlying CLI's default is cpu_count()).
    if args.nproc is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        from multiprocessing import cpu_count as _cpu_count
        args.nproc = min(8, int(slurm_cpus) if slurm_cpus else _cpu_count())
        print(f"[info] --nproc resolved to {args.nproc} "
              f"(from {'SLURM_CPUS_PER_TASK' if slurm_cpus else 'cpu_count()'})")
    elif args.nproc == 0:
        args.nproc = None  # sentinel: let CLI use its own default

    # Per-mol docking timeout — propagated as env var read by selectbench.data.dock.
    # Overrides any prior SELECTBENCH_DOCK_TIMEOUT so --timeout is authoritative.
    os.environ["SELECTBENCH_DOCK_TIMEOUT"] = str(args.timeout)

    if args.sdf:
        # Explicit SDF list: --method must be a single name (results need somewhere to land).
        if args.method is None or len(args.method) != 1:
            sys.exit("ERROR: --sdf requires exactly one --method <name> (so the "
                     "results dir is unambiguous).")
        methods_runs = [(args.method[0], args.sdf)]
    else:
        if args.method is None:
            args.method = auto_discover_methods(kind)
            print(f"[info] auto-discovered methods: {args.method}")
        if not args.method:
            sys.exit(f"No methods found under generated_samples/*/{kind}/")
        if args.results_name and len(args.method) > 1:
            sys.exit("ERROR: --results_name is incompatible with multiple methods.")
        methods_runs = []
        for method in args.method:
            files = _discover(method, kind)
            if not files:
                print(f"[skip] {method}: no .sdf or .smi under "
                      f"generated_samples/{method}/{kind}/")
                continue
            methods_runs.append((method, files))

    if not methods_runs:
        sys.exit("Nothing to evaluate.")

    failures = []
    for method, input_files in methods_runs:
        run_one_method(method, input_files, kind, args, known_case_studies,
                       known_pois, failures)

    cmd_failures = [(rc, c) for rc, c in failures if rc is not None]
    skipped_inputs = [c for rc, c in failures if rc is None]
    if cmd_failures:
        print(f"\n[ERROR] {len(cmd_failures)} evaluation command(s) failed:")
        for rc, cmd in cmd_failures:
            print(f"  (exit {rc}) {' '.join(cmd)}")
    if skipped_inputs:
        print(f"\n[ERROR] {len(skipped_inputs)} input file(s) were not evaluated "
              f"because the filename matched no known target:")
        for fp in skipped_inputs:
            print(f"  {fp}")
    if cmd_failures or skipped_inputs:
        sys.exit(1)
    print("Done.")


def run_one_method(method, input_files, kind, args, known_case_studies, known_pois,
                   failures=None):
    print(f"=================== {method} ===================")
    print(f"Found {len(input_files)} file(s): {[os.path.basename(s) for s in input_files]}")

    results_name = args.results_name or method
    results_base = args.results_dir or os.path.join(REPO_ROOT, "results")
    out_dir = os.path.join(results_base, results_name, kind)
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {os.path.join(results_base, results_name, kind)}/\n")

    # --- Case study evaluation ---
    if args.eval == "cs":
        if args.out_csv and len(input_files) > 1:
            print(f"  [SKIP] --out_csv is set but {len(input_files)} input file(s) "
                  f"were found; refusing to overwrite a single output. Pass one file "
                  f"(via --sdf) or drop --out_csv to use per-file names.")
            return
        for input_file in input_files:
            ext = os.path.splitext(input_file)[1].lower()
            file_flag = "--sdf_file" if ext == ".sdf" else "--smiles_file"
            case_study = args.case_study or infer_case_study(input_file, known_case_studies)
            print(f"=== {os.path.basename(input_file)} (case_study={case_study or 'none'}) ===")
            if case_study is None:
                print(f"  [SKIP] {os.path.basename(input_file)}: does not match any "
                      f"known case study ({', '.join(known_case_studies)}); "
                      f"not evaluated (use --case_study to specify).")
                # Same contract as the safety branch: an input that was never
                # evaluated is a failure, not a success. rc=None distinguishes
                # "never ran" from "ran and failed"; both exit non-zero.
                if failures is not None:
                    failures.append((None, input_file))
            else:
                stem = os.path.splitext(os.path.basename(input_file))[0]
                out_csv = os.path.join(out_dir, args.out_csv or f"{stem}.csv")
                cmd = ["selectbench", "evaluate-case",
                       "--case_study", case_study,
                       file_flag, input_file,
                       "--output", out_csv]
                if args.nproc:
                    cmd += ["--nproc", str(args.nproc)]
                if args.exhaustiveness is not None:
                    cmd += ["--exhaustiveness", str(args.exhaustiveness)]
                if args.no_fallback:
                    cmd += ["--no-fallback"]
                run_cmd(cmd, args.dry_run, failures)
            print()

    # --- Safety target evaluation ---
    if args.eval == "safety":
        # Build {poi_name: [matching files]}. One PoI may have many files
        # (compose runs add a sweep+mode suffix), so we iterate every match.
        files_by_poi = {}
        unmatched = []
        for fp in input_files:
            poi_match = infer_case_study(fp, known_pois)
            if poi_match:
                files_by_poi.setdefault(poi_match, []).append(fp)
            else:
                unmatched.append(fp)

        # Report per file rather than only when *nothing* matched: a single
        # mistyped filename would otherwise be dropped in silence and the run
        # would still exit 0, looking like a success.
        for fp in unmatched:
            print(f"  [SKIP] {os.path.basename(fp)}: does not match any known PoI "
                  f"({', '.join(known_pois)}); not evaluated.")
            if failures is not None:
                # rc=None marks "input never evaluated", distinct from a command
                # that ran and failed; both make the run exit non-zero.
                failures.append((None, fp))

        pois = args.poi if args.poi else sorted(files_by_poi)
        if not pois:
            print("[SKIP] safety eval: no input files match known POIs "
                  f"({', '.join(known_pois)}). Use --poi to specify.")

        total_jobs = sum(len(files_by_poi.get(p, [])) for p in pois)
        if args.out_csv and total_jobs > 1:
            print(f"  [SKIP] --out_csv is set but {total_jobs} input file(s) "
                  f"were matched across {len(pois)} POI(s); refusing to overwrite a "
                  f"single output. Pass one file (via --sdf) or drop --out_csv to use "
                  f"per-file names.")
            return

        for poi in pois:
            matches = files_by_poi.get(poi, [])
            if not matches:
                print(f"=== {poi} ===")
                print(f"  [SKIP] no matching SDF/SMI file found for POI '{poi}'")
                print()
                continue

            for matched_file in matches:
                ext = os.path.splitext(matched_file)[1].lower()
                file_flag = "--sdf_file" if ext == ".sdf" else "--smiles_file"
                stem = os.path.splitext(os.path.basename(matched_file))[0]
                print(f"=== {os.path.basename(matched_file)} (poi={poi}) ===")

                out_csv = os.path.join(out_dir, args.out_csv or f"{stem}.csv")
                cmd = ["selectbench", "evaluate-safety",
                       file_flag, matched_file,
                       "--output", out_csv,
                       "--poi", poi]
                if args.priority_targets:
                    cmd += ["--priority_targets"]
                if args.targets:
                    for t in args.targets:
                        cmd += ["--targets", t]
                if args.nproc:
                    cmd += ["--nproc", str(args.nproc)]
                if args.exhaustiveness is not None:
                    cmd += ["--exhaustiveness", str(args.exhaustiveness)]
                if args.no_fallback:
                    cmd += ["--no-fallback"]
                run_cmd(cmd, args.dry_run, failures)
                print()

    # --- CrossDocked evaluation ---
    if args.eval == "crossdocked":
        case_info, on_index, pair_index = load_crossdocked_indices()

        case_ids_override = args.case_ids_parsed
        if case_ids_override is not None:
            unknown = [c for c in case_ids_override if c not in case_info]
            if unknown:
                sys.exit(f"ERROR: --case_ids not present in crossdocked pairs CSV: "
                         f"{unknown} (valid range 0..{max(case_info)}).")
            print(f"[info] --case_ids override: {case_ids_override}")

        for input_file in input_files:
            stem = os.path.splitext(os.path.basename(input_file))[0]
            jobs = (
                [(cid, f"{case_info[cid][0]}_vs_{case_info[cid][1]}__"
                       f"{case_info[cid][2]}.csv")
                 for cid in case_ids_override]
                if case_ids_override is not None
                else crossdocked_jobs_for(stem, on_index, pair_index)
            )
            print(f"=== {os.path.basename(input_file)} (jobs={len(jobs)}) ===")
            if not jobs:
                print(f"  [SKIP] could not infer case_ids from stem '{stem}'. "
                      f"Expected '<on>__<diff>', "
                      f"'<on>_vs_<off>__<diff>[_<sweep>]', or an on-target "
                      f"pocket name (e.g. '3tym_A'); not evaluated.")
                # See the cs/safety branches: a never-evaluated input exits
                # non-zero rather than looking like a clean run.
                if failures is not None:
                    failures.append((None, input_file))
                print()
                continue

            ext = os.path.splitext(input_file)[1].lower()
            file_flag = "--sdf_file" if ext == ".sdf" else "--smiles_file"

            for case_id, default_name in jobs:
                if args.out_csv and len(jobs) > 1:
                    print(f"  [SKIP] --out_csv is set but stem '{stem}' fans out "
                          f"to {len(jobs)} case_ids; refusing to overwrite.")
                    break
                out_csv = os.path.join(out_dir, args.out_csv or default_name)
                cmd = ["selectbench", "evaluate-xl",
                       "--case_id", str(case_id),
                       file_flag, input_file,
                       "--output", out_csv]
                if args.nproc:
                    cmd += ["--nproc", str(args.nproc)]
                if args.exhaustiveness is not None:
                    cmd += ["--exhaustiveness", str(args.exhaustiveness)]
                if args.no_fallback:
                    cmd += ["--no-fallback"]
                run_cmd(cmd, args.dry_run, failures)
            print()


if __name__ == "__main__":
    main()
