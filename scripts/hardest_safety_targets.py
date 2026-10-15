"""
For each (method, PoI), rank the safety panel targets by how hard they are to
distinguish from the on-target (i.e., how *un*selective the method is against
them). Writes one CSV per method (for programmatic use — e.g. negative
guidance config) plus a combined CSV for overview.

"Hardest" = target against which the method is least selective.
Primary criterion: lowest selectivity_rate_1kcal_<target>  (fewest molecules with Δ ≤ -1)
Tiebreaks:      : lowest selectivity_rate_<target>, then highest (least-negative) delta_mean_<target>

Two ranking *scopes* are emitted side by side (a `scope` column distinguishes
them):
  - `all`      — ranked over the full safety panel (84 anti-targets)
  - `priority` — ranked over the priority subset only (22 anti-targets)
Ranks restart at 1 within each (method, PoI, scope).

Outputs:
  results/<method>/safety/hardest_safety_targets.csv  (cols: scope, poi, rank, target, sr_1kcal, sr, delta_mean)
  results/hardest_safety_targets.csv                  (all methods; adds a `method` column)

To pick the top-N hardest targets for a given (method, PoI, scope) in
downstream code, filter rows where scope matches and rank <= N.

By default the script scans results/<method>/safety/<poi>.csv. Pass
`--from NAME=DIR` (repeatable) to instead rank from an external directory of
safety CSVs, labelled as method NAME — DIR may use either the nested
`safety/<poi>.csv` layout or a flat `safety_<poi>.csv` layout, e.g.

  python scripts/hardest_safety_targets.py \
      --from diffsbdd=/path/to/results_safety_exh1_correct_box/diffsbdd
"""

import argparse
import csv
import os
import glob
import re
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from selectbench.metrics import (  # noqa: E402
    safety_metrics,
    load_raw_rows,
    SAFETY_TARGETS,
    SAFETY_TARGET_PRIORITY,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# (scope label) -> (panel subset). `all` is ranked first so it carries the
# full safety_metrics computation; `priority` reuses it (see rank_scopes).
SCOPES = {
    "all": SAFETY_TARGETS,
    "priority": SAFETY_TARGET_PRIORITY,
}


def rank_scopes_from_raw(csv_path):
    """Compute per-target selectivity once over the full panel, then return
    {scope: [target dicts hardest-first]} for every scope in SCOPES.

    Per-target selectivity rates are independent of which other targets share
    the panel, so a single safety_metrics() call over the full list serves
    every scope — each scope is just a filtered, re-sorted view of it."""
    rows = load_raw_rows(csv_path)
    if not rows:
        return {}
    m = safety_metrics(rows, SAFETY_TARGETS)

    def ranked(targets):
        per_target = []
        for t in targets:
            sr_1k = m.get(f"selectivity_rate_1kcal_{t}")
            if sr_1k is None:
                continue
            per_target.append({
                "target": t,
                "sr_1kcal": sr_1k,
                "sr": m.get(f"selectivity_rate_{t}"),
                "delta_mean": m.get(f"delta_{t}"),
            })
        per_target.sort(key=lambda x: (
            x["sr_1kcal"],
            x["sr"] if x["sr"] is not None else float("inf"),
            -(x["delta_mean"] if x["delta_mean"] is not None else float("-inf")),
        ))
        return per_target

    return {scope: ranked(targets) for scope, targets in SCOPES.items()}


def format_row(scope, rank, entry, poi=None, method=None):
    row = {
        "scope": scope,
        "poi": poi,
        "rank": rank,
        "target": entry["target"],
        "sr_1kcal": f"{entry['sr_1kcal']:.4f}",
        "sr": f"{entry['sr']:.4f}" if entry["sr"] is not None else "",
        "delta_mean": f"{entry['delta_mean']:.4f}" if entry["delta_mean"] is not None else "",
    }
    if method is not None:
        row = {"method": method, **row}
    return row


# Compose runs write sweep-tagged stems next to the plain <poi>.csv, e.g.
# `parp1_2off_priority_bon2.5_boff-0.5_FKC.csv`. Those are not PoIs; treating
# them as such multiplies the emitted ranking and makes the cross-method
# overview unreadable. Mirrors cross_method._discover_targets' skip logic.
#
# Both compose filename forms must be covered. gen_compose_*.py emits
# `bon<x>_boff<y>` when either --beta_on/--beta_off is given, and falls back to
# `w<guidance_weight>` when neither is -- so a default-guidance run produces
# `parp1_w1.0_FKC.csv`, which an alternation missing the w-form would rank as a
# PoI named "parp1_w1.0_FKC". Numbers use the same character class as
# cross_method._SWEEP_LABEL_RE so scientific notation (`bon1e-2`) matches too.
_SWEEP_STEM_RE = re.compile(r"_\d+off_|_bon[\d.eE+-]+_boff|_w[\d.eE+-]+_")
_SKIP_STEMS = {"hardest_safety_targets", "summary"}


def discover_safety_files():
    """Default mode: {method: {poi: csv_path}} from results/<method>/safety/<poi>.csv.

    Skips the tool's own output, the `summary/` tree, and compose sweep stems.
    """
    found = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "safety", "*.csv"))):
        method = os.path.basename(os.path.dirname(os.path.dirname(p)))
        poi = os.path.splitext(os.path.basename(p))[0]
        if method in _SKIP_STEMS or poi in _SKIP_STEMS:
            continue
        if _SWEEP_STEM_RE.search(poi):
            continue
        found[method][poi] = p
    return found


def safety_files_in(directory):
    """{poi: csv_path} for a directory holding safety CSVs in either the nested
    `safety/<poi>.csv` layout or a flat `safety_<poi>.csv` layout."""
    out = {}
    candidates = (glob.glob(os.path.join(directory, "safety", "*.csv"))
                  + glob.glob(os.path.join(directory, "safety_*.csv")))
    for p in sorted(candidates):
        name = os.path.splitext(os.path.basename(p))[0]
        if name == "hardest_safety_targets":
            continue
        poi = name[len("safety_"):] if name.startswith("safety_") else name
        out[poi] = p
    return out


def parse_from(spec):
    """Parse a `NAME=DIR` --from spec."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--from expects NAME=DIR (got {spec!r})")
    name, directory = spec.split("=", 1)
    name, directory = name.strip(), os.path.expanduser(directory.strip())
    if not name:
        raise argparse.ArgumentTypeError(f"--from: empty method name in {spec!r}")
    if not os.path.isdir(directory):
        raise argparse.ArgumentTypeError(f"--from: not a directory: {directory}")
    return name, directory


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="sources", action="append", default=[],
                        type=parse_from, metavar="NAME=DIR",
                        help="Rank from an external dir of safety CSVs labelled "
                             "as method NAME (repeatable). Overrides the default "
                             "results/<method>/safety scan.")
    args = parser.parse_args()

    if args.sources:
        # method -> {poi: csv_path}
        files_by_method = defaultdict(dict)
        for name, directory in args.sources:
            found = safety_files_in(directory)
            if not found:
                print(f"  [warn] no safety CSVs under {directory}")
            files_by_method[name].update(found)
    else:
        files_by_method = discover_safety_files()

    # method -> list of rows (all PoIs, all scopes, all ranks)
    by_method = defaultdict(list)
    combined = []

    for method, poi_files in sorted(files_by_method.items()):
        for poi, path in sorted(poi_files.items()):
            scoped = rank_scopes_from_raw(path)
            if not scoped:
                print(f"  [warn] no rows in {path}")
                continue
            for scope in SCOPES:
                for rank, entry in enumerate(scoped.get(scope, []), start=1):
                    by_method[method].append(format_row(scope, rank, entry, poi=poi))
                    combined.append(format_row(scope, rank, entry, poi=poi, method=method))

    per_method_fields = ["scope", "poi", "rank", "target", "sr_1kcal", "sr", "delta_mean"]
    # Sort: PoI alphabetical, then scope, then rank ascending
    sort_key = lambda r: (r["poi"], r["scope"], r["rank"])
    for method, rows in sorted(by_method.items()):
        out_path = os.path.join(RESULTS_DIR, method, "safety",
                                "hardest_safety_targets.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        rows.sort(key=sort_key)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_method_fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {method:14s} -> {os.path.relpath(out_path, REPO_ROOT)}  ({len(rows)} rows)")

    if not combined:
        print("No safety CSVs found — nothing written.")
        return

    combined_fields = ["method"] + per_method_fields
    combined.sort(key=lambda r: (r["method"], r["poi"], r["scope"], r["rank"]))
    combined_path = os.path.join(RESULTS_DIR, "hardest_safety_targets.csv")
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=combined_fields)
        writer.writeheader()
        writer.writerows(combined)
    print(f"\nCombined: {os.path.relpath(combined_path, REPO_ROOT)}  ({len(combined)} rows)")


if __name__ == "__main__":
    main()
