"""Summarize Boltz2 affinity results across all case studies and methods."""

import argparse
from pathlib import Path

import pandas as pd

CASE_DISPLAY_NAMES = {
    "parp": "PARP1 (on) / PARP2 (off)",
    "diffsbdd_kinase": "BMP2K (on) / TTK (off)",
    "cdk2_cdk9_jcim": "CDK9 (on) / CDK2 (off)",
    "cdk2_erk2_jcim": "ERK2 (on) / CDK2 (off)",
    "drd2_hrh1_jmed": "DRD2 (on) / HRH1 (off)",
}

MODEL_ORDER = [
    "decompdiff",
    "diffsbdd",
    "targetdiff",
    "molform",
    "molsnapper",
    "pocket2mol",
    "pocketflow",
    "cmdgen",
    "speclig",
    "decompdiff_cfg",
    "decompdiff_fkc",
    "diffsbdd_cfg",
    "diffsbdd_fkc",
    "targetdiff_cfg",
    "targetdiff_fkc",
]

MODEL_DISPLAY_NAMES = {
    "decompdiff": "DecompDiff",
    "diffsbdd": "DiffSBDD",
    "targetdiff": "TargetDiff",
    "molform": "MolForm",
    "molsnapper": "MolSnapper",
    "pocket2mol": "Pocket2Mol",
    "pocketflow": "PocketFlow",
    "cmdgen": "CMD-GEN",
    "speclig": "SpecLig",
    "decompdiff_cfg": "DecompDiff + CFG",
    "decompdiff_fkc": "DecompDiff + FKC",
    "diffsbdd_cfg": "DiffSBDD + CFG",
    "diffsbdd_fkc": "DiffSBDD + FKC",
    "targetdiff_cfg": "TargetDiff + CFG",
    "targetdiff_fkc": "TargetDiff + FKC",
}


def load_results(results_dir: Path, cap: int | None = 100) -> pd.DataFrame:
    """Walk the ``results_dir / <model> / cs / <case>.csv`` layout written by
    ``scripts/run_boltz2_case_study.py`` and stack the per-mol CSVs into one
    long-form DataFrame keyed by ``(case_study, model)``. If ``cap`` is set,
    take the first ``cap`` rows per (model, case) and print a note when any
    file gets truncated.
    """
    rows = []
    truncated = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        cs_dir = model_dir / "cs"
        if not cs_dir.is_dir():
            continue
        for csv_path in sorted(cs_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df["case_study"] = csv_path.stem
            df["model"] = model_dir.name
            if cap is not None and len(df) > cap:
                truncated.append((model_dir.name, csv_path.stem, len(df), cap))
                df = df.head(cap)
            rows.append(df)
    if not rows:
        raise ValueError(
            f"No CSVs found under {results_dir}. Expected the layout "
            f"{results_dir}/<model>/cs/<case>.csv written by "
            f"scripts/run_boltz2_case_study.py."
        )
    for model, case, n, k in truncated:
        print(f"  [cap] {model}/cs/{case}.csv: kept first {k} of {n} rows")
    return pd.concat(rows, ignore_index=True)


def save_top_delta_molecules(df: pd.DataFrame, output_dir: Path, n: int = 10) -> None:
    for (case_study, model), group in df.groupby(["case_study", "model"]):
        group = group.copy()
        group["delta_score"] = group["on_target_affinity_prob"] - group["off_target_affinity_prob"]
        top = group.nlargest(n, "delta_score")[["mol_idx", "smiles", "on_target_affinity_prob", "off_target_affinity_prob", "delta_score"]]
        case_dir = output_dir / case_study
        case_dir.mkdir(parents=True, exist_ok=True)
        top.to_csv(case_dir / f"{model}.csv", index=False)
    print(f"\nSaved top-{n} delta molecules to {output_dir}")


def save_selective_molecules(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for (case_study, model), group in df.groupby(["case_study", "model"]):
        rank_on = group["on_target_affinity_prob"].rank(pct=True)
        rank_off = group["off_target_affinity_prob"].rank(pct=True)
        mask = (rank_on > 0.6) & (rank_off < 0.4)
        selective = group[mask][["mol_idx", "smiles", "on_target_affinity_prob", "off_target_affinity_prob"]].copy()
        case_dir = output_dir / case_study
        case_dir.mkdir(parents=True, exist_ok=True)
        selective.to_csv(case_dir / f"{model}.csv", index=False)
    print(f"\nSaved selective molecules to {output_dir}")


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["delta_score"] = df["on_target_affinity_prob"] - df["off_target_affinity_prob"]
    records = []
    for (case_study, model), group in df.groupby(["case_study", "model"]):
        delta = group["delta_score"].dropna()
        rank_on = group["on_target_affinity_prob"].rank(pct=True)
        rank_off = group["off_target_affinity_prob"].rank(pct=True)
        rank_delta = (rank_on - rank_off).dropna()
        record = {
            "case_study": case_study,
            "model": model,
            "n_mols": len(group),
            "frac_rank_delta_gt0": (rank_delta > 0).mean(),
            "frac_rank_on_gt06_off_lt04": ((rank_on > 0.6) & (rank_off < 0.4)).mean(),
            "frac_on_gt06_off_lt04": ((group["on_target_affinity_prob"] > 0.6) & (group["off_target_affinity_prob"] < 0.4)).mean(),
            "mean_on_target": group["on_target_affinity_prob"].dropna().mean(),
            # Population std (ddof=0) to match cross_method._boltz2_metrics and
            # metrics._pstd — pandas .std() defaults to sample std (ddof=1).
            "std_on_target": group["on_target_affinity_prob"].dropna().std(ddof=0),
            "mean_delta": delta.mean(),
            "std_delta": delta.std(ddof=0),
            "top5_mean_delta": delta.nlargest(5).mean(),
            "top10_mean_delta": delta.nlargest(10).mean(),
        }
        records.append(record)
    result = pd.DataFrame(records)
    model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    result["_order"] = result["model"].map(model_rank)
    result = result.sort_values(["case_study", "_order"]).drop(columns="_order")
    result["model"] = result["model"].map(MODEL_DISPLAY_NAMES).fillna(result["model"])
    return result


def to_latex_table(summary: pd.DataFrame) -> pd.DataFrame:
    fmt = "{:.3f}"
    latex = summary[["case_study", "model"]].copy()
    latex["mean_on_target"] = summary.apply(
        lambda r: rf"${fmt.format(r['mean_on_target'])}_{{\pm {fmt.format(r['std_on_target'])}}}$", axis=1
    )
    latex["mean_delta"] = summary.apply(
        lambda r: rf"${fmt.format(r['mean_delta'])}_{{\pm {fmt.format(r['std_delta'])}}}$", axis=1
    )
    for col in ["top5_mean_delta", "top10_mean_delta", "frac_rank_delta_gt0", "frac_on_gt06_off_lt04", "frac_rank_on_gt06_off_lt04"]:
        latex[col] = summary[col].map(lambda v: f"${fmt.format(v)}$")
    return latex


SECTION_HEADERS = {
    "DecompDiff": "Standard SBDD models",
    "CMD-GEN": "Selectivity-trained models",
    "DecompDiff + CFG": "Inference-time guided methods",
}


def write_latex_table(latex: pd.DataFrame, output_path: Path) -> None:
    data_cols = [c for c in latex.columns if c != "case_study"]
    n_cols = len(data_cols)
    header = " & ".join(data_cols)
    lines = [
        "{\\tiny",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\renewcommand{\\arraystretch}{1.00}",
        "\\begin{longtable}{lccccccc}",
        "\\mycaption{\\datasetA  evaluation}{}",
        "\\label{tab:case_study_boltz2} \\\\",
        "\\toprule",
        "Method &",
        "\\makecell{$\\bar{s}_{\\text{on}}$\\\\$(\\uparrow)$} &",
        "\\makecell{$\\bar{\\Delta}$\\\\$(\\uparrow)$} &",
        "\\makecell{$\\bar{\\Delta}_{\\text{top-}5}$\\\\$(\\uparrow)$} &",
        "\\makecell{$\\bar{\\Delta}_{\\text{top-}10}$\\\\$(\\uparrow)$} &",
        "\\makecell{$\\mathrm{SR}^{\\mathrm{rank}}_{\\tau=0}$\\\\$(\\uparrow)$} &",
        "\\makecell{$\\mathrm{SR}_{s_{\\text{on}}>0.6, s_{\\text{off}}<0.4}$\\\\$(\\uparrow)$} &",
        "\\makecell{$\\mathrm{SR}^{\\mathrm{rank}}_{p_{\\text{on}}>0.6, p_{\\text{off}}<0.4}$\\\\$(\\uparrow)$}   \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Method &",
        "\\makecell{$\\bar{s}_{\\text{on}}$\\\\$(\\uparrow)$} &",
        "\\makecell{$\\bar{\\Delta}$\\\\$(\\uparrow)$} &",
        "\\makecell{$\\bar{\\Delta}_{\\text{top-}5}$\\\\$(\\uparrow)$} &",
        "\\makecell{$\\bar{\\Delta}_{\\text{top-}10}$\\\\$(\\uparrow)$} &",
        "\\makecell{$\\mathrm{SR}^{\\mathrm{rank}}_{\\tau=0}$\\\\$(\\uparrow)$} &",
        "\\makecell{$\\mathrm{SR}_{s_{\\text{on}}>0.6, s_{\\text{off}}<0.4}$\\\\$(\\uparrow)$} &",
        "\\makecell{$\\mathrm{SR}^{\\mathrm{rank}}_{p_{\\text{on}}>0.6, p_{\\text{off}}<0.4}$\\\\$(\\uparrow)$}   \\\\",
        "\\midrule",
        "\\endhead",
        f"\\multicolumn{{{n_cols}}}{{r}}{{Continued on next page}} \\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    colors = ["Blue1", "Pink1", "Yellow2", "Purple1", "Blue2"]
    for i, (case_study, group) in enumerate(latex.groupby("case_study", sort=False)):
        case_display = CASE_DISPLAY_NAMES.get(case_study, case_study)
        color = colors[i % len(colors)]
        if i > 0:
            lines.append("\\midrule")
        inner = f"\\hlcustom{{{color}}}{{\\texttt{{Case: {case_display}}}}}"
        lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{{inner}}} \\\\")
        lines.append("\\midrule")
        for row in group[data_cols].itertuples(index=False):
            model = row[0]
            if model in SECTION_HEADERS:
                section = SECTION_HEADERS[model]
                if model != "DecompDiff":
                    lines.append("\\midrule")
                lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{{section}}}}} \\\\")
                lines.append("\\midrule")
            lines.append(" & ".join(str(v) for v in row) + " \\\\")
    lines.append("\\end{longtable}}")
    output_path.write_text("\n".join(lines) + "\n")


REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results_dir", type=Path, default=REPO_ROOT / "results" / "boltz2",
        help="Directory laid out as <results_dir>/<model>/cs/<case>.csv (written by scripts/run_boltz2_case_study.py).",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to write the per-(case, model) summary CSV.")
    parser.add_argument("--latex_output", type=str, default=None,
                        help="Opt-in: path to write a LaTeX longtable of the summary. "
                             "The table uses SelectBench paper-specific macros "
                             "(\\mycaption, \\datasetA) and will not compile "
                             "standalone. Omit to skip.")
    parser.add_argument("--selective_output_dir", type=Path, default=None,
                        help="Opt-in: directory to write per-(case, model) CSVs of "
                             "molecules ranking high on-target and low off-target. "
                             "Omit to skip.")
    parser.add_argument("--top_delta_output_dir", type=Path, default=None,
                        help="Opt-in: directory to write per-(case, model) CSVs of the "
                             "top-10 molecules by affinity delta. Omit to skip.")
    parser.add_argument("--cap", type=int, default=100,
                        help="Cap each (model, case) at this many rows (0 disables).")
    args = parser.parse_args()

    cap = args.cap if args.cap and args.cap > 0 else None
    df = load_results(args.results_dir, cap=cap)
    if args.selective_output_dir:
        save_selective_molecules(df, args.selective_output_dir)
    if args.top_delta_output_dir:
        save_top_delta_molecules(df, args.top_delta_output_dir)
    summary = compute_metrics(df)

    print(summary.to_string(index=False))

    if args.output:
        summary.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")

    if args.latex_output:
        latex = to_latex_table(summary)
        latex_path = Path(args.latex_output)
        write_latex_table(latex, latex_path)
        print(f"\nSaved LaTeX table to {latex_path}")


if __name__ == "__main__":
    main()
