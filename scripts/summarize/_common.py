"""Shared helpers for cross_method.py.

Centralised here to keep entry-point scripts thin and consistent. Anything that loads
a file path or augments per-mol rows in-memory belongs in this module.
"""

import csv
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from selectbench.metrics import (  # noqa: E402,F401
    cs_metrics, safety_metrics, load_raw_rows, SAFETY_TARGETS, SAFETY_TARGET_PRIORITY,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")
CASE_STUDIES_CSV = os.path.join(REPO_ROOT, "data", "case_studies", "case_studies.csv")
CROSSDOCKED_CSV = os.path.join(REPO_ROOT, "data", "crossdocked_test_set",
                               "crossdocked_test_set_pairs.csv")

_ZSCORE_DIR = os.path.join(REPO_ROOT, "data", "zscore_library", "processed")
DEFAULT_CS_STATS = os.path.join(_ZSCORE_DIR, "case_studies_stats.csv")
# Safety eval needs PoI keys (standard_docking_benchmark_stats.csv) AND
# safety-panel target keys (safety_targets_stats.csv). Defaults to the OS
# pathsep-joined pair so load_reference_stats merges them.
DEFAULT_SAFETY_STATS = os.pathsep.join([
    os.path.join(_ZSCORE_DIR, "standard_docking_benchmark_stats.csv"),
    os.path.join(_ZSCORE_DIR, "safety_targets_stats.csv"),
])
DEFAULT_XD_STATS = os.path.join(_ZSCORE_DIR, "crossdocked_test_set_stats.csv")


def fmt(v):
    """Format a metric for a summary CSV cell; None/non-numeric become "".

    Counts (n_valid, n_total, n_scored, ...) are ints and are written as ints --
    rendering a molecule count as ``100.0000`` reads like a measurement rather
    than a tally. Bools are excluded because ``bool`` is a subclass of ``int``.
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v is None:
        return ""
    return str(v) if isinstance(v, int) else f"{v:.4f}"


def load_case_pdb_keys(csv_path=CASE_STUDIES_CSV):
    """case_name -> (on_key, off_key); each key is ``{pdb_id}_{chain_id}``."""
    keys = {}
    if not os.path.isfile(csv_path):
        return keys
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            on = r["on_target"]
            p1 = f"{r['pdb1_id']}_{r['pdb1_chain_id']}"
            p2 = f"{r['pdb2_id']}_{r['pdb2_chain_id']}"
            on_key, off_key = (p1, p2) if on == "pdb1" else (p2, p1)
            keys[r["case_name"]] = (on_key, off_key)
    return keys


def load_crossdocked_pair_info(csv_path=CROSSDOCKED_CSV):
    """case_id -> (on_key, off_key, difficulty)."""
    info = {}
    if not os.path.isfile(csv_path):
        return info
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            on = r["on_target"]
            p1 = f"{r['pdb1_id']}_{r['pdb1_chain_id']}"
            p2 = f"{r['pdb2_id']}_{r['pdb2_chain_id']}"
            on_key, off_key = (p1, p2) if on == "pdb1" else (p2, p1)
            info[int(r["case_id"])] = (on_key, off_key, r["difficulty"])
    return info


def load_reference_stats(csv_path):
    """target_id -> (mean, std). Skips rows with non-numeric/empty stats.

    Accepts a single path or an OS-pathsep-joined list of paths (e.g. the
    safety default merges PoI stats + safety-target stats). Files later in
    the list override earlier ones on key collision.
    """
    stats = {}
    if not csv_path:
        return stats
    for p in csv_path.split(os.pathsep):
        if not p or not os.path.isfile(p):
            continue
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    stats[r["target_id"]] = (float(r["mean_score"]), float(r["std_score"]))
                except (TypeError, ValueError):
                    continue
    return stats


def _zscore(score, mean_std):
    if mean_std is None or score in (None, ""):
        return None
    mean, std = mean_std
    if std == 0:
        return None
    try:
        return (float(score) - mean) / std
    except (TypeError, ValueError):
        return None


def augment_cs_rows_with_zscores(rows, on_key, off_key, ref_stats):
    """Add zscore_on_target, zscore_off_target, delta_z to each row in place."""
    on_ms = ref_stats.get(on_key)
    off_ms = ref_stats.get(off_key)
    if on_ms is None or off_ms is None:
        return
    for r in rows:
        z_on = _zscore(r.get("score_on_target"), on_ms)
        z_off = _zscore(r.get("score_off_target"), off_ms)
        r["zscore_on_target"] = z_on
        r["zscore_off_target"] = z_off
        r["delta_z"] = (z_on - z_off) if (z_on is not None and z_off is not None) else None


def augment_safety_rows_with_zscores(rows, on_target_id, safety_target_names, ref_stats):
    """Add zscore_on_target, zscore_<t>, mean_zscore_safety, delta_z_mean,
    is_top20_z to each row in place. Mirrors the eval-time logic in
    selectbench.cli (use_z=True path)."""
    on_ms = ref_stats.get(on_target_id)
    if on_ms is None:
        return
    for r in rows:
        z_on = _zscore(r.get("score_on_target"), on_ms)
        r["zscore_on_target"] = z_on
        z_safety = []
        for t in safety_target_names:
            zt = _zscore(r.get(f"score_{t}"), ref_stats.get(t))
            r[f"zscore_{t}"] = zt
            if zt is not None:
                z_safety.append(zt)
        mzs = (sum(z_safety) / len(z_safety)) if z_safety else None
        r["mean_zscore_safety"] = mzs
        r["delta_z_mean"] = (z_on - mzs) if (z_on is not None and mzs is not None) else None
        if z_on is not None and z_safety:
            all_z = sorted([z_on] + z_safety)
            thr_idx = max(0, int(len(all_z) * 0.2) - 1)
            r["is_top20_z"] = int(z_on <= all_z[thr_idx])
        else:
            r["is_top20_z"] = None


def cap_to_n_valid(rows, valid_field, cap):
    """Return the prefix of ``rows`` containing at most ``cap`` rows whose
    ``valid_field`` parses as a float. Invalid rows in that prefix are kept
    so ``n_valid/n_total`` still reflects docking-success rate."""
    if cap is None:
        return rows
    valid = 0
    out = []
    for r in rows:
        out.append(r)
        v = r.get(valid_field)
        if v not in (None, ""):
            try:
                float(v)
                valid += 1
            except (TypeError, ValueError):
                pass
        if valid >= cap:
            break
    return out


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
