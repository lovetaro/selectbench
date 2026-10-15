#!/usr/bin/env python3
"""Tier-3 summary — one CSV per (case | poi | crossdocked target), rows = methods.

Reads raw per-mol CSVs from the new layout:

    results/<method>/cs/<case>.csv
    results/<method>/safety/<poi>.csv
    results/<method>/crossdocked/<on>_vs_<off>__<diff>[_<sweep>].csv

and writes cross-method comparison tables:

    --mode cs           → results/summary/cs/<case>.csv
    --mode safety       → results/summary/safety/<poi>.csv
    --mode crossdocked  → results/summary/crossdocked/<diff>.csv
                          (pair-weighted within each difficulty tier; one row per method)
    --mode all          → all three (default)

Compose methods (any results dir whose name ends in ``_compose``) are split into
synthetic per-mode rows like ``diffsbdd_compose_cfgsde`` / ``diffsbdd_compose_fkc``.
For each (mode, case), the file with the best primary metric (default: ``delta``)
is picked when multiple sweep points exist.

Score functions (``--score_fn``):
    docking  (default) — uses Vina docking scores from ``results/<method>/``.
    boltz2             — reads ``results/boltz2/<model>/cs/<case>.csv``; one row
                         per model; writes ``results/summary/boltz2/cs/<case>.csv``.
                         Only ``--mode cs`` is supported for boltz2.

Usage:
    python scripts/summarize/cross_method.py                  # cs + safety + crossdocked
    python scripts/summarize/cross_method.py --mode cs
    python scripts/summarize/cross_method.py --mode crossdocked
    python scripts/summarize/cross_method.py --score_fn boltz2
"""

import argparse
import csv
import glob
import os
import re

from _common import (  # noqa: E402
    REPO_ROOT, RESULTS_DIR, SUMMARY_DIR,
    DEFAULT_CS_STATS, DEFAULT_SAFETY_STATS, DEFAULT_XD_STATS,
    fmt, load_case_pdb_keys, load_crossdocked_pair_info, load_reference_stats,
    augment_cs_rows_with_zscores, augment_safety_rows_with_zscores,
    cap_to_n_valid, write_csv,
    cs_metrics, safety_metrics, load_raw_rows, SAFETY_TARGETS, SAFETY_TARGET_PRIORITY,
)

COMPOSE_MODES = ["CFG-SDE", "FKC"]  # filename suffix → synthetic method tag

# Boltz2 (score_fn=boltz2) reads/writes a separate results/boltz2 tree so it
# never collides with the docking outputs.
BOLTZ2_RESULTS_DIR = os.path.join(RESULTS_DIR, "boltz2")
BOLTZ2_SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary", "boltz2")

# Multi off-target compose variants. None = legacy 1-off (filename has no
# ``<K>off_<scope>`` infix); 2/3 = multi-guidance runs with K off-targets and
# a ``<scope>`` (priority|all) tag. β is swept the same way as the 1-off panel
# (wildcard match + best-by-delta), not fixed to a single point — see the
# compose filename convention in scripts/generate_compose/.
COMPOSE_K_OFF_VARIANTS = [None, 2, 3]

# Per-mol CSVs store delta as `score_on - score_off` (legacy: lower = more
# selective). User-facing summary tables flip to `score_off - score_on`
# (higher = more selective) AND rename each flipped column to make the
# convention explicit — a `delta_*` column in a summary CSV is renamed to
# `delta_selectivity_*` so it never collides with the raw-CSV `delta`.
#
# The map below is old_summary_key → renamed_summary_key. Values are also
# negated (except _std columns, which are sign-invariant).
_DELTA_RENAME_MAP = {
    # CS + XL
    "delta":                    "delta_selectivity",
    "delta_std":                "delta_selectivity_std",
    "delta_mean_1kcal":         "delta_selectivity_mean_1kcal",
    "delta_mean_1kcal_std":     "delta_selectivity_mean_1kcal_std",
    "top5_delta_mean":          "top5_delta_selectivity_mean",
    "top10_delta_mean":         "top10_delta_selectivity_mean",
    "delta_z":                  "delta_selectivity_z",
    "delta_z_std":              "delta_selectivity_z_std",
    # Safety
    "delta_mean":               "delta_selectivity_mean",
    "delta_mean_std":           "delta_selectivity_mean_std",
    "worst_delta_mean":         "worst_delta_selectivity_mean",
    "worst_delta_mean_std":     "worst_delta_selectivity_mean_std",
    "delta_z_mean":             "delta_selectivity_z_mean",
    "delta_z_mean_std":         "delta_selectivity_z_mean_std",
}
_STD_KEYS = {k for k in _DELTA_RENAME_MAP if k.endswith("_std")}


def _apply_sign_convention(m: dict, target_names=None) -> dict:
    """Return a copy of ``m`` with delta keys renamed to ``delta_selectivity``
    variants (see ``_DELTA_RENAME_MAP``) and their values negated so positive =
    more selective. Per-target ``delta_<t>`` columns also get renamed and
    flipped when ``target_names`` is supplied. ``_std`` columns are renamed
    only (sign-invariant)."""
    out = dict(m)
    for old, new in _DELTA_RENAME_MAP.items():
        if old not in out:
            continue
        v = out.pop(old)
        out[new] = -v if (v is not None and old not in _STD_KEYS) else v
    for t in (target_names or []):
        old = f"delta_{t}"
        if old in out:
            v = out.pop(old)
            out[f"delta_selectivity_{t}"] = -v if v is not None else v
    return out


def auto_discover_methods(results_dir=RESULTS_DIR):
    """Return every results/<name>/ subdir that holds at least one of the
    expected eval buckets (cs/, safety/, crossdocked/). Skips ``summary/``.

    Lets users drop a new method into ``results/<new_method>/`` and have it
    picked up by ``cross_method.py`` without editing this file.
    """
    if not os.path.isdir(results_dir):
        return []
    found = []
    for name in sorted(os.listdir(results_dir)):
        if name == "summary":
            continue
        path = os.path.join(results_dir, name)
        if not os.path.isdir(path):
            continue
        if any(os.path.isdir(os.path.join(path, k))
               for k in ("cs", "safety", "crossdocked")):
            found.append(name)
    return found

# Column order for the cs output table. Columns absent from every method's
# computed metrics are dropped before write. `delta_selectivity_*` are the
# sign-flipped renames from `_apply_sign_convention` (positive = selective).
CS_COLUMNS = [
    # Molecule counts first: every metric below is conditioned on them, and a
    # thin cell (few reconstructable molecules) is otherwise invisible here.
    "n_valid", "n_total",
    "score_on_target", "score_on_target_std",
    "score_off_target", "delta_selectivity", "delta_selectivity_std",
    "selectivity_rate", "selectivity_rate_1kcal",
    "delta_selectivity_mean_1kcal", "delta_selectivity_mean_1kcal_std",
    "top5_delta_selectivity_mean", "top10_delta_selectivity_mean",
    # z-score variants (optional)
    "zscore_on_target", "zscore_on_target_std",
    "zscore_off_target",
    "delta_selectivity_z", "delta_selectivity_z_std",
    "selectivity_rate_z", "selectivity_rate_z_1",
]

# In the crossdocked tier tables these are summed across pairs (tier totals);
# every other column is averaged. See _xd_by_difficulty_table.
XD_SUMMED_COLUMNS = {"n_valid", "n_total"}

SAFETY_COLUMNS_AGGREGATE = [
    "n_valid", "n_total",
    "score_on_target", "score_on_target_std",
    "mean_safety_score",
    "delta_selectivity_mean", "delta_selectivity_mean_std",
    "worst_delta_selectivity_mean", "worst_delta_selectivity_mean_std",
    "strict_selectivity_rate", "strict_selectivity_rate_1kcal",
    "selectivity_rate_mean", "selectivity_rate_1kcal_mean",
    "avg_selectivity_rate", "avg_selectivity_rate_1kcal",
    "min_selectivity_rate", "min_selectivity_rate_1kcal",
    "ratio20",
    # z-score variants
    "zscore_on_target", "zscore_on_target_std",
    "mean_zscore_safety",
    "delta_selectivity_z_mean", "delta_selectivity_z_mean_std",
    "avg_selectivity_rate_z", "min_selectivity_rate_z",
    "avg_selectivity_rate_z_1", "min_selectivity_rate_z_1",
    "ratio20_z",
]

# Reference-free molecular-quality columns, appended to a table only under
# --quality (opt-in). Computed from the `smiles` column already in every raw
# CSV — see selectbench.sbdd_metrics. Not delta metrics, so they are exempt from
# the sign convention (higher qed/sa/diversity is better; logp/mw/lipinski are
# descriptive). In the crossdocked table these are averaged across pairs in a
# difficulty tier, like every other numeric column there.
QUALITY_COLUMNS = [
    "n_scored",
    "qed_mean", "qed_std",
    "sa_mean", "sa_std",
    "logp_mean", "logp_std",
    "lipinski_mean", "lipinski_std",
    "mw_mean", "mw_std",
    "diversity",
]


def _quality_for_rows(rows):
    """Compute the molecular-quality aggregate for one file's raw rows.

    Reads the always-present `smiles` column and returns a dict restricted to
    QUALITY_COLUMNS. RDKit is imported lazily here so the default (selectivity-
    only) summarize path never requires it.
    """
    from selectbench.sbdd_metrics import compute_quality_from_smiles
    smiles = [r.get("smiles", "") for r in rows]
    q = compute_quality_from_smiles(smiles)
    return {c: q.get(c) for c in QUALITY_COLUMNS}


# ── Method resolution ───────────────────────────────────────────────────────

def _expand_methods(methods):
    """Turn ``["diffsbdd", "diffsbdd_compose"]`` into:

        [("diffsbdd",                        "diffsbdd",         None,     None),
         ("diffsbdd_compose_cfgsde",         "diffsbdd_compose", "CFG-SDE", None),
         ("diffsbdd_compose_fkc",            "diffsbdd_compose", "FKC",     None),
         ("diffsbdd_compose_2off_cfgsde",    "diffsbdd_compose", "CFG-SDE", 2),
         ("diffsbdd_compose_2off_fkc",       "diffsbdd_compose", "FKC",     2),
         ("diffsbdd_compose_3off_cfgsde",    "diffsbdd_compose", "CFG-SDE", 3),
         ("diffsbdd_compose_3off_fkc",       "diffsbdd_compose", "FKC",     3)]

    where each tuple is (display_name, results_dir, mode_filter, k_off).
    Compose dirs are auto-split per COMPOSE_MODES x COMPOSE_K_OFF_VARIANTS;
    everything else passes through with mode=None, k_off=None.
    """
    expanded = []
    for m in methods:
        if m.endswith("_compose"):
            for k_off in COMPOSE_K_OFF_VARIANTS:
                for mode in COMPOSE_MODES:
                    tag = mode.lower().replace("-", "")
                    if k_off is None:
                        display = f"{m}_{tag}"
                    else:
                        display = f"{m}_{k_off}off_{tag}"
                    expanded.append((display, m, mode, k_off))
        else:
            expanded.append((m, m, None, None))
    return expanded


def _glob_matches(results_dir, kind, target, mode, k_off=None, scope=None):
    """Return ALL per-mol CSVs for one (display method, target) pair, sorted.

    Baseline (mode=None): the single exact ``<target>.csv`` (list of 0 or 1).
    Compose 1-off (mode=MODE, k_off=None):
        every ``<target>_bon*_boff*_<mode>.csv`` sweep point, plus the
        default-guidance form ``<target>_w*_<mode>.csv`` (written by
        gen_compose_*.py when neither --beta_on nor --beta_off is passed).
        Excludes multi-off filenames (those start with ``<target>_<K>off_<scope>_``).
    Compose multi-off (mode=MODE, k_off in {2,3}, scope in {"priority","all"}):
        every ``<target>_<K>off_<scope>_bon*_boff*_<mode>.csv`` and
        ``<target>_<K>off_<scope>_w*_<mode>.csv`` sweep point. The ``<scope>``
        infix is written by gen_compose_*.py so the priority / all safety panels
        each pick up the matching hardest-target run.
    """
    base = os.path.join(results_dir, kind)
    if mode is None:
        path = os.path.join(base, f"{target}.csv")
        return [path] if os.path.isfile(path) else []
    if k_off is None:
        # Legacy 1-off compose: <target>_bon<x>_boff<y>_<mode>.csv and the
        # default-guidance <target>_w<w>_<mode>.csv (no <K>off_<scope> infix).
        return sorted(
            glob.glob(os.path.join(base, f"{target}_bon*_boff*_{mode}.csv"))
            + glob.glob(os.path.join(base, f"{target}_w*_{mode}.csv")))
    # Multi-off: scope-tagged, β or default-guidance swept like the 1-off panel.
    return sorted(
        glob.glob(os.path.join(base, f"{target}_{k_off}off_{scope}_bon*_boff*_{mode}.csv"))
        + glob.glob(os.path.join(base, f"{target}_{k_off}off_{scope}_w*_{mode}.csv")))


def _glob_match(results_dir, kind, target, mode, k_off=None, scope=None,
                delta_key="delta"):
    """Find the single best per-mol CSV for one (display method, target) pair.

    Baseline (mode=None) returns the exact file. Compose returns the sweep
    point with the smallest mean delta (most selective) via
    ``_pick_best_by_delta``; see ``_glob_matches`` for the filename schemas.
    ``delta_key`` must name a column the per-mol CSVs of this ``kind``
    actually carry -- ``delta`` for cs/crossdocked, ``delta_mean`` for safety.
    """
    matches = _glob_matches(results_dir, kind, target, mode, k_off=k_off, scope=scope)
    if not matches:
        return None
    if mode is None:
        return matches[0]
    return _pick_best_by_delta(matches, delta_key=delta_key)


_SWEEP_LABEL_RE = re.compile(r"_(bon[\d.eE+-]+_boff[\d.eE+-]+|w[\d.eE+-]+)_")


def _sweep_label(path):
    """Extract the β/weight sweep token (e.g. ``bon2.5_boff-0.5`` or ``w1.0``)
    from a compose filename, for labelling per-sweep rows under --sweep_detail.
    Falls back to the bare stem when no token is found."""
    m = _SWEEP_LABEL_RE.search(os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


MIN_SWEEP_COVERAGE = 0.8


def _pick_best_by_delta(paths, delta_key="delta", min_frac=MIN_SWEEP_COVERAGE):
    """For multiple compose sweep points on the same (method, target), pick
    the file whose mean delta is smallest (most selective).

    ``delta_key`` names the per-molecule column holding that delta. The cs and
    crossdocked per-mol CSVs carry a bare ``delta``; the safety CSVs do not --
    they carry ``delta_<target>`` per anti-target plus an aggregate
    ``delta_mean``. Passing the wrong key makes every candidate unparseable and
    silently degrades this into "return the alphabetically first sweep point",
    so the fallback below warns instead of failing quietly.

    Candidates are first restricted to sweep points whose valid-molecule count
    is at least ``min_frac`` of the best-covered point. Without that guard a
    setting whose reconstruction mostly failed wins on a small-sample mean: a
    point yielding 3 molecules produces a far more extreme average than one
    yielding 95, and "most selective" then measures generation failure rather
    than selectivity. When every point is equally thin the guard is a no-op and
    all of them stay eligible -- that is a data gap, not something selection
    can repair, so it is reported rather than hidden.
    """
    stats = {}
    for p in paths:
        deltas = []
        for r in load_raw_rows(p):
            try:
                deltas.append(float(r.get(delta_key, "")))
            except (TypeError, ValueError):
                continue
        if deltas:
            stats[p] = (len(deltas), sum(deltas) / len(deltas))
    if not stats:
        print(f"  [warn] no numeric '{delta_key}' in any of {len(paths)} sweep "
              f"point(s) for {os.path.basename(paths[0])}; falling back to the "
              f"first by filename order")
        return paths[0]

    n_max = max(n for n, _ in stats.values())
    eligible = {p: v for p, v in stats.items() if v[0] >= min_frac * n_max}
    best = min(eligible, key=lambda p: eligible[p][1])

    unguarded = min(stats, key=lambda p: stats[p][1])
    if unguarded is not best:
        print(f"  [sweep] {os.path.basename(best)} (n={stats[best][0]}) chosen over "
              f"{os.path.basename(unguarded)} (n={stats[unguarded][0]}, "
              f"mean {stats[unguarded][1]:+.2f}): the latter falls below "
              f"{min_frac:.0%} of the best sweep point's {n_max} molecules")
    return best


# ── CS table ────────────────────────────────────────────────────────────────

def _compute_cs_table(methods, case, case_keys=None, ref_stats=None,
                      cap=None, results_dir=RESULTS_DIR, sweep_detail=False,
                      quality=False):
    """Return (header, rows) where each row is one method's CS metrics.

    With ``sweep_detail`` the compose sweep is not collapsed to best-by-delta:
    every sweep point becomes its own row, labelled ``<method>_<sweep>``.
    """
    on_off = case_keys.get(case) if case_keys else None
    rows = []
    for display, src_dir, mode, k_off in _expand_methods(methods):
        if k_off is not None:
            # Multi-off rows are a safety-panel concept; no CS-side files.
            continue
        method_dir = os.path.join(results_dir, src_dir)
        if sweep_detail:
            paths = _glob_matches(method_dir, "cs", case, mode)
        else:
            best = _glob_match(method_dir, "cs", case, mode)
            paths = [best] if best else []
        for path in paths:
            data = load_raw_rows(path)
            data = cap_to_n_valid(data, "delta", cap)
            if ref_stats and on_off:
                augment_cs_rows_with_zscores(data, on_off[0], on_off[1], ref_stats)
            m = cs_metrics(data)
            if m.get("n_valid"):
                label = (f"{display}_{_sweep_label(path)}"
                         if sweep_detail and mode is not None else display)
                mm = _apply_sign_convention(m)
                if quality:
                    mm.update(_quality_for_rows(data))
                rows.append((label, mm))

    col_order = CS_COLUMNS + (QUALITY_COLUMNS if quality else [])
    cols_used = [c for c in col_order
                 if any(r[1].get(c) is not None for r in rows)]
    header = ["method"] + cols_used
    out_rows = [
        [d] + [fmt(m.get(c)) for c in cols_used]
        for d, m in rows
    ]
    return header, out_rows


# ── Safety table ────────────────────────────────────────────────────────────

def _build_safety_table(rows, target_names, quality=False):
    """Format ``rows`` (list of (display, metrics_dict)) for ``target_names``."""
    agg_order = SAFETY_COLUMNS_AGGREGATE + (QUALITY_COLUMNS if quality else [])
    agg_cols = [c for c in agg_order
                if any(r[1].get(c) is not None for r in rows)]
    per_target = []
    for t in target_names:
        for k in (f"delta_selectivity_{t}",
                  f"selectivity_rate_{t}",
                  f"selectivity_rate_1kcal_{t}"):
            if any(r[1].get(k) is not None for r in rows):
                per_target.append(k)
    header = ["method"] + agg_cols + per_target
    out_rows = [
        [d] + [fmt(m.get(c)) for c in agg_cols] + [fmt(m.get(c)) for c in per_target]
        for d, m in rows
    ]
    return header, out_rows


def _covers_panel(path, target_names, scope):
    """True if the per-mol safety CSV at ``path`` was docked against every
    target in ``target_names``.

    One file normally feeds both the all-target and the priority panel, which
    assumes it was docked against the full panel. A run made with
    ``evaluate-safety --priority_targets`` covers only the 22 priority targets;
    scoring it on the 84-target panel would silently report priority-panel
    numbers as all-panel ones. Coverage is read from the ``score_<target>``
    columns in the header, so empty cells from failed dockings don't count
    against it.
    """
    with open(path, newline="") as f:
        header = next(csv.reader(f), [])
    docked = {c[len("score_"):] for c in header
              if c.startswith("score_") and c != "score_on_target"}
    n_missing = sum(1 for t in target_names if t not in docked)
    if n_missing:
        print(f"  [panel] {os.path.basename(path)}: not used for the {scope} "
              f"panel -- docked against {len(target_names) - n_missing} of its "
              f"{len(target_names)} targets")
    return n_missing == 0


def _compute_safety_table(methods, poi, ref_stats=None, cap=None,
                          results_dir=RESULTS_DIR, sweep_detail=False,
                          quality=False):
    """Return two (header, rows) pairs: one for all targets, one for priority targets.

    Each scope panel is built from files matching that scope: multi-off rows
    use ``<poi>_<K>off_<scope>_...`` and so a row's priority-panel numbers come
    from the priority-scope run (which guided against the top-K priority
    anti-targets), while its all-panel numbers come from the all-scope run.
    Non-multi-off rows ignore the scope arg and share one file across panels.

    With ``sweep_detail`` the compose sweep is not collapsed to best-by-delta:
    every sweep point becomes its own row, labelled ``<method>_<sweep>``.
    """
    def _build_rows(scope, target_names):
        rows = []
        for display, src_dir, mode, k_off in _expand_methods(methods):
            method_dir = os.path.join(results_dir, src_dir)
            matches = _glob_matches(method_dir, "safety", poi, mode,
                                    k_off=k_off, scope=scope)
            # Coverage is checked before the sweep pick, so a priority-only
            # sweep point can never win the all-panel selection.
            paths = [p for p in matches if _covers_panel(p, target_names, scope)]
            if paths and not sweep_detail:
                paths = [paths[0] if mode is None
                         else _pick_best_by_delta(paths, delta_key="delta_mean")]
            for path in paths:
                data = load_raw_rows(path)
                data = cap_to_n_valid(data, "delta_mean", cap)
                if ref_stats:
                    augment_safety_rows_with_zscores(data, poi, SAFETY_TARGETS, ref_stats)
                m = safety_metrics(data, target_names)
                if m.get("n_valid"):
                    label = (f"{display}_{_sweep_label(path)}"
                             if sweep_detail and mode is not None else display)
                    mm = _apply_sign_convention(m, target_names)
                    if quality:
                        mm.update(_quality_for_rows(data))
                    rows.append((label, mm))
        return rows

    all_rows = _build_rows("all", SAFETY_TARGETS)
    priority_rows = _build_rows("priority", SAFETY_TARGET_PRIORITY)

    # `custom`-scope compose runs (--safety_target a b c) are docked but cannot
    # be aggregated: the filename records only "custom", not which targets were
    # used, so there is no panel to score them against. Say so instead of
    # dropping them silently — the GPU + docking time is already spent.
    custom = sorted({os.path.basename(p) for p in
                     glob.glob(os.path.join(results_dir, "*", "safety",
                                            "*off_custom_*.csv"))})
    if custom:
        print(f"[warn] {len(custom)} custom-scope safety file(s) are not "
              f"aggregated (the panel they used is not recoverable from the "
              f"filename). Use --hardest_scope priority|all for runs you intend "
              f"to summarize:")
        for name in custom:
            print(f"    {name}")

    return (
        _build_safety_table(all_rows, SAFETY_TARGETS, quality=quality),
        _build_safety_table(priority_rows, SAFETY_TARGET_PRIORITY, quality=quality),
    )


# ── Crossdocked tables ──────────────────────────────────────────────────────

# Match files like "<on>_vs_<off>__<diff>[_<sweep>].csv" inside the
# crossdocked/ subdir. <on>/<off> are <pdb>_<chain>.
_XD_FILE_RE = re.compile(
    r"^(?P<on>[^_]+_[^_]+)_vs_(?P<off>[^_]+_[^_]+)__(?P<diff>easy|medium|hard)"
    r"(?:_(?P<sweep>.+))?\.csv$"
)


def _parse_xd_filename(path):
    """Return (on, off, diff, sweep) for a crossdocked CSV, or None."""
    m = _XD_FILE_RE.match(os.path.basename(path))
    if not m:
        return None
    return m["on"], m["off"], m["diff"], m["sweep"]


def _compute_one_pair_metrics(path, pair_info_lookup, ref_stats, cap,
                              quality=False):
    """Load one xd CSV → (case_id, metrics_dict) or (None, None)."""
    parsed = _parse_xd_filename(path)
    if parsed is None:
        return None, None
    on, off, diff, _sweep = parsed
    cid = pair_info_lookup.get((on, off, diff))
    if cid is None:
        return None, None
    data = load_raw_rows(path)
    data = cap_to_n_valid(data, "delta", cap)
    if ref_stats:
        augment_cs_rows_with_zscores(data, on, off, ref_stats)
    m = cs_metrics(data)
    if not m.get("n_valid"):
        return None, None
    mm = _apply_sign_convention(m)
    if quality:
        mm.update(_quality_for_rows(data))
    return cid, mm


def _xd_by_difficulty_table(methods, difficulty, pair_info, ref_stats=None,
                            cap=None, case_ids_filter=None,
                            results_dir=RESULTS_DIR, quality=False):
    """For --mode crossdocked: pair-weighted one-row-per-method per tier.

    Returns (header, rows) or (None, None) when nothing matches.
    """
    pair_lookup = {(on, off, diff): cid
                   for cid, (on, off, diff) in pair_info.items()}
    tier_pairs = {(on, off, diff): cid
                  for cid, (on, off, diff) in pair_info.items()
                  if diff == difficulty
                  and (case_ids_filter is None or cid in case_ids_filter)}
    if not tier_pairs:
        return None, None

    per_method_metrics = {}   # display -> [pair_metrics]

    for display, src_dir, mode, k_off in _expand_methods(methods):
        if k_off is not None:
            # Multi-off rows are a safety-panel concept; no XD-side files.
            continue
        xd_dir = os.path.join(results_dir, src_dir, "crossdocked")
        if not os.path.isdir(xd_dir):
            continue
        per_pair_files = {}
        for path in sorted(glob.glob(os.path.join(xd_dir, "*.csv"))):
            parsed = _parse_xd_filename(path)
            if not parsed:
                continue
            on, off, diff, sweep = parsed
            if (on, off, diff) not in tier_pairs:
                continue
            if mode is None and sweep is not None:
                continue
            if mode is not None and sweep is None:
                continue
            if mode is not None and not sweep.endswith(mode):
                continue
            per_pair_files.setdefault((on, off, diff), []).append(path)

        for key, paths in per_pair_files.items():
            chosen = _pick_best_by_delta(paths) if len(paths) > 1 else paths[0]
            cid, metrics = _compute_one_pair_metrics(chosen, pair_lookup,
                                                     ref_stats, cap,
                                                     quality=quality)
            if cid is None:
                continue
            per_method_metrics.setdefault(display, []).append(metrics)

    if not per_method_metrics:
        return None, None

    col_order = CS_COLUMNS + (QUALITY_COLUMNS if quality else [])
    cols_used = [c for c in col_order
                 if any(any(m.get(c) is not None for m in pms)
                        for pms in per_method_metrics.values())]

    def defined(c, pms):
        return [m[c] for m in pms if m.get(c) is not None]

    # Tier rows average each metric over the tier's pairs, with two exceptions.
    # Molecule counts are summed: a tier total, not a per-pair mean. And some
    # metrics are undefined for a pair -- delta_selectivity_mean_1kcal has no
    # value when no molecule in that pair clears 1 kcal/mol -- so their mean
    # runs over fewer pairs than n_pairs. Averaging over the defined pairs is the
    # only meaningful choice (there is no value to substitute), but it must not
    # be silent: such a metric gets a companion <metric>_n_pairs column holding
    # its actual denominator. A _std column always shares its base metric's
    # pairs (both come from the same per-molecule list), so one companion
    # follows the pair.
    partial = {c for c in cols_used
               if c not in XD_SUMMED_COLUMNS and not c.endswith("_std")
               and any(len(defined(c, pms)) < len(pms)
                       for pms in per_method_metrics.values())}
    layout = []  # (output column, source metric, how to aggregate)
    for c in cols_used:
        how = "sum" if c in XD_SUMMED_COLUMNS else "mean"
        layout.append((c, c, how))
        base = c[:-len("_std")] if c.endswith("_std") else c
        family_end = (base + "_std") if (base + "_std") in cols_used else base
        if base in partial and c == family_end:
            layout.append((base + "_n_pairs", base, "count"))

    header = ["method", "n_pairs"] + [name for name, _, _ in layout]
    rows = []
    for display, pms in per_method_metrics.items():
        row = [display, str(len(pms))]
        for _name, src, how in layout:
            vals = defined(src, pms)
            if how == "count":
                row.append(str(len(vals)))
            elif how == "sum":
                row.append(fmt(sum(vals)) if vals else "")
            else:
                row.append(fmt(sum(vals) / len(vals)) if vals else "")
        rows.append(row)
    return header, rows


# ── Boltz2 score function ───────────────────────────────────────────────────

BOLTZ2_CS_COLUMNS = [
    "n_mols",
    "mean_on_target", "mean_on_target_std",
    "mean_delta", "mean_delta_std",
    "top5_mean_delta", "top10_mean_delta",
    "frac_rank_delta_gt0",
    "frac_on_gt06_off_lt04",
    "frac_rank_on_gt06_off_lt04",
]


def _pct_rank(vals):
    """Return average percentile ranks (0–1) for each value, ties averaged."""
    n = len(vals)
    if n == 0:
        return []
    sorted_idx = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and vals[sorted_idx[j + 1]] == vals[sorted_idx[j]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg
        i = j + 1
    return [r / n for r in ranks]


def _boltz2_metrics(rows):
    """Compute boltz2 selectivity metrics from per-mol rows.

    Expects rows with ``on_target_affinity_prob`` and
    ``off_target_affinity_prob`` columns.
    """
    on_probs, off_probs = [], []
    for r in rows:
        try:
            on_probs.append(float(r["on_target_affinity_prob"]))
            off_probs.append(float(r["off_target_affinity_prob"]))
        except (KeyError, TypeError, ValueError):
            continue
    n = len(on_probs)
    if n == 0:
        return {}

    deltas = [on - off for on, off in zip(on_probs, off_probs)]
    sorted_deltas_desc = sorted(deltas, reverse=True)

    rank_on = _pct_rank(on_probs)
    rank_off = _pct_rank(off_probs)
    rank_deltas = [ron - roff for ron, roff in zip(rank_on, rank_off)]

    mean_on = sum(on_probs) / n
    mean_delta = sum(deltas) / n
    std_on = (sum((x - mean_on) ** 2 for x in on_probs) / n) ** 0.5
    std_delta = (sum((x - mean_delta) ** 2 for x in deltas) / n) ** 0.5

    top5 = sorted_deltas_desc[:5]
    top10 = sorted_deltas_desc[:10]

    return {
        "n_mols": n,
        "mean_on_target": mean_on,
        "mean_on_target_std": std_on,
        "mean_delta": mean_delta,
        "mean_delta_std": std_delta,
        "top5_mean_delta": sum(top5) / len(top5) if top5 else None,
        "top10_mean_delta": sum(top10) / len(top10) if top10 else None,
        "frac_rank_delta_gt0": sum(1 for d in rank_deltas if d > 0) / n,
        "frac_on_gt06_off_lt04": sum(
            1 for on, off in zip(on_probs, off_probs) if on > 0.6 and off < 0.4
        ) / n,
        "frac_rank_on_gt06_off_lt04": sum(
            1 for ron, roff in zip(rank_on, rank_off) if ron > 0.6 and roff < 0.4
        ) / n,
    }


def _discover_boltz2_methods(results_dir):
    """Return sorted list of method names (subdirs of results_dir containing cs/)."""
    if not os.path.isdir(results_dir):
        return []
    return sorted(
        name for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name, "cs"))
    )


def _discover_boltz2_cases(results_dir):
    """Return sorted union of case CSV stems across all <method>/cs/ subdirs."""
    cases = set()
    for method in _discover_boltz2_methods(results_dir):
        cs_dir = os.path.join(results_dir, method, "cs")
        for f in os.listdir(cs_dir):
            if f.endswith(".csv"):
                cases.add(os.path.splitext(f)[0])
    return sorted(cases)


def _compute_boltz2_cs_table(case, results_dir, cap=None):
    """Return (header, rows) for one boltz2 case study — one row per model."""
    method_rows = []
    for method in _discover_boltz2_methods(results_dir):
        path = os.path.join(results_dir, method, "cs", f"{case}.csv")
        if not os.path.isfile(path):
            continue
        data = load_raw_rows(path)
        data = cap_to_n_valid(data, "on_target_affinity_prob", cap)
        m = _boltz2_metrics(data)
        if m.get("n_mols"):
            method_rows.append((method, m))

    cols_used = [c for c in BOLTZ2_CS_COLUMNS
                 if any(r[1].get(c) is not None for r in method_rows)]
    header = ["method"] + cols_used
    out_rows = [
        [d] + [fmt(m.get(c)) for c in cols_used]
        for d, m in method_rows
    ]
    return header, out_rows


# ── Discovery ──────────────────────────────────────────────────────────────

def _discover_targets(methods, kind, results_dir=RESULTS_DIR):
    """Find the union of <name> stems under results/<m>/<kind>/<name>.csv
    across all expanded methods. Compose dirs are stripped of the trailing
    sweep+mode tag so multiple sweep points collapse to one target name.
    """
    SKIP_STEMS = {"hardest_safety_targets", "summary"}
    names = set()
    seen_dirs = set()
    for _display, src_dir, mode, _k_off in _expand_methods(methods):
        if (src_dir, mode) in seen_dirs:
            continue  # k_off variants share a results_dir/mode glob
        seen_dirs.add((src_dir, mode))
        base = os.path.join(results_dir, src_dir, kind)
        for p in sorted(glob.glob(os.path.join(base, "*.csv"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem in SKIP_STEMS:
                continue
            # `<poi>_priority` is the name of the priority-panel summary. A
            # results file with that stem would become a target of its own
            # whose all-panel summary is written to the same path as <poi>'s
            # priority summary, overwriting it.
            if kind == "safety" and stem.endswith("_priority"):
                print(f"  [skip] {p}: `_priority` is reserved for summary "
                      f"tables. Save a priority-only run as "
                      f"{stem[:-len('_priority')]}.csv; only its priority "
                      f"panel is summarized.")
                continue
            if mode is None:
                names.add(stem)
                continue
            # Skip legacy multi-off compose files written without a scope tag
            # (e.g. parp1_3off_bon2.5_boff-0.5_FKC) — those predate the
            # priority/all panel split and are not used here.
            if re.search(r"_\d+off_(?!priority|all)", stem):
                continue
            # strip trailing "[_<K>off_<scope>]_<sweep>_<MODE>" where
            # <sweep> is either "bon<x>_boff<y>" or "w<w>" (the two
            # filename schemas written by gen_compose_*.py); the
            # <K>off_<scope> infix is present only for multi-guidance.
            sweep = r"(?:bon[\d.eE+-]+_boff[\d.eE+-]+|w[\d.eE+-]+)"
            multi = r"(?:_\d+off_(?:priority|all))?"
            m = re.match(r"^(?P<target>.+?)" + multi + r"_" + sweep + r"_"
                         + re.escape(mode) + r"$", stem)
            if m:
                names.add(m.group("target"))
    return sorted(names)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--score_fn", choices=["docking", "boltz2"], default="docking",
                   help="Score function: 'docking' (default) uses Vina scores; "
                        "'boltz2' uses Boltz2 affinity probabilities. boltz2 reads "
                        "results/boltz2/<model>/cs/<case>.csv and writes to "
                        "results/summary/boltz2/cs/<case>.csv. Only --mode cs is "
                        "supported for boltz2.")
    p.add_argument("--mode", choices=["cs", "safety", "crossdocked", "all"],
                   default="all",
                   help="'cs', 'safety', 'crossdocked': single benchmark. "
                        "'all' (default): cs + safety + crossdocked. "
                        "Crossdocked output is pair-weighted per difficulty tier. "
                        "Ignored (treated as 'cs') when --score_fn boltz2.")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Method names. Compose dirs (ending in '_compose') "
                        f"auto-split into per-mode rows for each of {COMPOSE_MODES}. "
                        "Default: auto-discover every results/<name>/ subdir that "
                        "contains a cs/, safety/, or crossdocked/ bucket. "
                        "Ignored when --score_fn boltz2 (models are discovered "
                        "automatically from results/boltz2/<model>/cs/).")
    p.add_argument("--out_dir", default=None,
                   help=f"Output base dir. Defaults to {SUMMARY_DIR} for docking, "
                        f"{BOLTZ2_SUMMARY_DIR} for boltz2.")
    p.add_argument("--results_dir", default=None,
                   help=f"Input base dir. Defaults to {RESULTS_DIR} for docking, "
                        f"{BOLTZ2_RESULTS_DIR} for boltz2.")
    p.add_argument("--cs_reference_stats", default=DEFAULT_CS_STATS,
                   help="Stats CSV (target_id,mean_score,std_score) for case-study "
                        "z-score augmentation. Empty string disables.")
    p.add_argument("--safety_reference_stats", default=DEFAULT_SAFETY_STATS,
                   help="Stats CSV for safety panel z-scores. Empty disables.")
    p.add_argument("--crossdocked_reference_stats", default=DEFAULT_XD_STATS,
                   help="Stats CSV for CrossDocked z-scores. Empty disables.")
    p.add_argument("--cap_n_valid", type=int, default=100,
                   help="Cap each method's input at the prefix containing this "
                        "many valid rows before computing metrics (default: 100; "
                        "pass 0 to disable).")
    p.add_argument("--case_ids", type=int, nargs="+", default=None,
                   help="Restrict crossdocked aggregation to this set of "
                        "case_ids (intersected with each tier).")
    p.add_argument("--sweep_detail", action="store_true",
                   help="Also emit <name>_sweeps.csv beside each cs/safety "
                        "summary, expanding every compose β/weight sweep point "
                        "into its own row (labelled <method>_<sweep>) instead of "
                        "collapsing the sweep to the best-by-delta pick. The "
                        "default best-only files are still written unchanged. "
                        "Crossdocked tables are pair-aggregated and always use "
                        "best-by-delta, so they get no _sweeps.csv.")
    p.add_argument("--quality", action="store_true",
                   help="Also compute reference-free molecular-quality metrics "
                        "(QED, SA, LogP, Lipinski, MW, Diversity) from the SMILES "
                        "in each raw CSV and append them as extra columns. "
                        "Requires RDKit. Off by default, so selectivity-only "
                        "output is unchanged. (docking score_fn only.)")
    args = p.parse_args()

    cap = args.cap_n_valid if args.cap_n_valid and args.cap_n_valid > 0 else None

    # ── Boltz2 branch ────────────────────────────────────────────────────────
    if args.score_fn == "boltz2":
        results_dir = args.results_dir or BOLTZ2_RESULTS_DIR
        out_dir = args.out_dir or BOLTZ2_SUMMARY_DIR
        if args.mode not in ("cs", "all"):
            print(f"[warn] --score_fn boltz2 only supports --mode cs; "
                  f"ignoring --mode {args.mode}")
        cases = _discover_boltz2_cases(results_dir)
        if not cases:
            print(f"[warn] no case subdirs found under {results_dir}")
            return
        print(f"[info] boltz2 cases: {cases}")
        for case in cases:
            header, rows = _compute_boltz2_cs_table(case, results_dir, cap=cap)
            if not rows:
                continue
            out = os.path.join(out_dir, "cs", f"{case}.csv")
            write_csv(out, header, rows)
            print(f"Wrote {out}  ({len(rows)} models, {len(header)-1} metric cols)")
        return

    # ── Docking branch ──────────────────────────────────────────────────────
    results_dir = args.results_dir or RESULTS_DIR
    out_dir = args.out_dir or SUMMARY_DIR
    if args.methods is None:
        args.methods = auto_discover_methods(results_dir)
        print(f"[info] auto-discovered methods: {args.methods}")

    # Each `<name>_compose` fans out into up to 7 synthetic rows
    # (K=1, K=2off, K=3off × CFG-SDE/FKC). Print the expanded list so users
    # don't wonder where the extra rows in the summary came from.
    expanded = [display for display, _src, _mode, _k in _expand_methods(args.methods)]
    if len(expanded) != len(args.methods):
        print(f"[info] compose fan-out → {len(expanded)} rows: {expanded}")

    case_keys = load_case_pdb_keys()
    cs_ref_stats = load_reference_stats(args.cs_reference_stats)
    safety_ref_stats = load_reference_stats(args.safety_reference_stats)
    xd_ref_stats = load_reference_stats(args.crossdocked_reference_stats)

    if args.mode in ("cs", "all"):
        for case in _discover_targets(args.methods, "cs", results_dir):
            header, rows = _compute_cs_table(args.methods, case,
                                             case_keys=case_keys,
                                             ref_stats=cs_ref_stats, cap=cap,
                                             results_dir=results_dir,
                                             quality=args.quality)
            if rows:
                out = os.path.join(out_dir, "cs", f"{case}.csv")
                write_csv(out, header, rows)
                print(f"Wrote {out}  ({len(rows)} methods, {len(header)-1} metric cols)")
            if args.sweep_detail:
                s_header, s_rows = _compute_cs_table(
                    args.methods, case, case_keys=case_keys,
                    ref_stats=cs_ref_stats, cap=cap,
                    results_dir=results_dir, sweep_detail=True,
                    quality=args.quality)
                if s_rows:
                    out = os.path.join(out_dir, "cs", f"{case}_sweeps.csv")
                    write_csv(out, s_header, s_rows)
                    print(f"Wrote {out}  ({len(s_rows)} sweep rows)")

    if args.mode in ("safety", "all"):
        for poi in _discover_targets(args.methods, "safety", results_dir):
            (header, rows), (p_header, p_rows) = _compute_safety_table(
                args.methods, poi,
                ref_stats=safety_ref_stats,
                cap=cap,
                results_dir=results_dir,
                quality=args.quality,
            )
            if rows:
                out = os.path.join(out_dir, "safety", f"{poi}.csv")
                write_csv(out, header, rows)
                print(f"Wrote {out}  ({len(rows)} methods, {len(header)-1} metric cols)")
            if p_rows:
                out = os.path.join(out_dir, "safety", f"{poi}_priority.csv")
                write_csv(out, p_header, p_rows)
                print(f"Wrote {out}  ({len(p_rows)} methods, {len(p_header)-1} metric cols)")
            if args.sweep_detail:
                (s_header, s_rows), (sp_header, sp_rows) = _compute_safety_table(
                    args.methods, poi,
                    ref_stats=safety_ref_stats,
                    cap=cap,
                    results_dir=results_dir,
                    sweep_detail=True,
                    quality=args.quality,
                )
                if s_rows:
                    out = os.path.join(out_dir, "safety", f"{poi}_sweeps.csv")
                    write_csv(out, s_header, s_rows)
                    print(f"Wrote {out}  ({len(s_rows)} sweep rows)")
                if sp_rows:
                    out = os.path.join(out_dir, "safety", f"{poi}_priority_sweeps.csv")
                    write_csv(out, sp_header, sp_rows)
                    print(f"Wrote {out}  ({len(sp_rows)} sweep rows)")

    if args.mode in ("crossdocked", "all"):
        pair_info = load_crossdocked_pair_info()
        if not pair_info:
            print("[warn] crossdocked pairs CSV not found; skipping crossdocked summaries")
            return

        difficulties = sorted({d for _, _, d in pair_info.values()})
        case_ids_filter = set(args.case_ids) if args.case_ids else None
        for diff in difficulties:
            header, rows = _xd_by_difficulty_table(args.methods, diff, pair_info,
                                                   ref_stats=xd_ref_stats, cap=cap,
                                                   case_ids_filter=case_ids_filter,
                                                   results_dir=results_dir,
                                                   quality=args.quality)
            if not rows:
                print(f"[skip] {diff}: no crossdocked results for any method")
                continue
            out = os.path.join(out_dir, "crossdocked", f"{diff}.csv")
            write_csv(out, header, rows)
            print(f"Wrote {out}  ({len(rows)} methods, {len(header)-2} metric cols)")


if __name__ == "__main__":
    main()
