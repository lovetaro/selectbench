"""Metric computations for case-study and safety-panel evaluations.

Pure functions that operate on lists of per-molecule dicts (as produced by the
docking workers in ``cli.py``). Used by summarize scripts to build Tier 2 and
Tier 3 aggregates. The raw per-mol CSVs never contain these aggregates —
they're always computed on demand.

Conventions
-----------

**Sign convention (per-mol, raw CSVs):**
    ``delta = score_on_target − score_off_target``   (raw Vina scores; lower =
    stronger binding). So **``delta ≤ 0`` ⇒ selective** for the on-target, and
    ``selectivity_rate = P(delta ≤ 0)``.

    Summary CSVs written by ``scripts/summarize/cross_method.py`` invert this
    for readability — the flipped column is renamed to ``delta_selectivity``
    (positive = selective), so the two conventions never share a column name.

**Aggregation vocabulary (safety-panel):**
    - ``selectivity_rate_mean``     — SR computed on the *per-mol* Δ_mean
                                       (pool over targets first, then rate).
    - ``avg_selectivity_rate``      — mean of the *per-target* SRs
                                       (rate per target first, then mean).
    - ``strict_selectivity_rate``   — SR computed on each mol's *worst* Δ_t
                                       (must beat every target).
    - ``min_selectivity_rate``      — min of the per-target SRs.
    The four are all "SR-like" but measure different things — see the
    ``safety_metrics`` docstring for the full definitions.

**Case study (on vs. off):**
    Required per-mol keys: ``score_on_target``, ``score_off_target``, ``delta``
    Optional: ``zscore_on_target``, ``zscore_off_target``, ``delta_z``

**Safety panel (on vs. 84 safety targets):**
    Required per-mol keys:
        ``score_on_target``, ``score_<target>`` for each safety target,
        ``mean_safety_score``, ``delta_<target>`` for each,
        ``delta_mean``, ``is_top20``
    Optional: ``zscore_on_target``, ``zscore_<target>``, ``mean_zscore_safety``,
              ``delta_z_mean``, ``is_top20_z``

"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from statistics import pstdev
from typing import Iterable, Optional


_SAFETY_TARGETS_CSV = Path(__file__).parents[2] / "data" / "safety_targets" / "safety_targets.csv"

# Excluded from every safety_targets.csv load path. htr1b_agonist is the same
# protein (HTR1B) as the 5ht1b PoI, so it'd double-count as both an on- and
# off-target. Exposed so cli.py can apply the same filter to its own loads.
EXCLUDED_SAFETY_TARGETS = {"htr1b_agonist"}


def _load_safety_target_lists():
    """Read safety_targets.csv → (full_list, priority_list), with excluded
    entries stripped from both. Returns ([], []) with a warning if the CSV
    isn't reachable (usually a non-editable ``pip install .`` where the
    ``data/`` tree isn't shipped alongside the package)."""
    full, priority = [], []
    if not _SAFETY_TARGETS_CSV.is_file():
        import warnings
        warnings.warn(
            f"selectbench.metrics: safety_targets.csv not found at "
            f"{_SAFETY_TARGETS_CSV}. SAFETY_TARGETS will be empty and "
            f"safety_metrics() will not aggregate per-target scores. "
            f"This usually means the package was installed non-editably "
            f"and the sibling data/ tree is missing; install with "
            f"`pip install -e .` from the repo root, or set the working "
            f"directory to a repo checkout.",
            RuntimeWarning, stacklevel=2,
        )
        return full, priority
    with open(_SAFETY_TARGETS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            name = r.get("case_name")
            if not name or name in EXCLUDED_SAFETY_TARGETS:
                continue
            full.append(name)
            try:
                if int(r.get("priority_target", "0")) == 1:
                    priority.append(name)
            except (TypeError, ValueError):
                continue
    return full, priority


SAFETY_TARGETS, SAFETY_TARGET_PRIORITY = _load_safety_target_lists()


# ── generic helpers ──────────────────────────────────────────────────────────

def _mean(xs: Iterable[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _pstd(xs: Iterable[float]) -> Optional[float]:
    """Population stdev. Returns None if no valid values, 0.0 for n=1."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    return pstdev(xs)


def _rate(xs: Iterable[float], threshold: float) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(1 for v in xs if v <= threshold) / len(xs)


def _as_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # Reject non-finite (NaN/inf): NaN would slip past the |score| sanity
    # bound (abs(nan) > 100 is False), count as "valid", then crash pstdev.
    return f if math.isfinite(f) else None


def _col(rows, key):
    """Extract one column as a list of floats (or None)."""
    return [_as_float(r.get(key)) for r in rows]


def top20_threshold_idx(n: int) -> int:
    """Index into a length-``n`` *ascending*-sorted score list for the
    "top 20 %" threshold. Callers use ``score <= sorted[idx]`` to test membership.

    Single source of truth so both ``cli.py`` (per-mol ``is_top20`` at eval
    time) and this module (``ratio20`` recomputation at summary time) agree.
    """
    return max(0, int(n * 0.2) - 1)


# ── case-study metrics ───────────────────────────────────────────────────────

def cs_metrics(rows: list[dict]) -> dict:
    """Given per-mol CS rows, return a flat dict of aggregate metrics.

    Keys: ``n``, ``n_valid``, ``score_on_target``, ``score_off_target``,
    ``delta``, ``selectivity_rate``, ``selectivity_rate_1kcal``,
    ``delta_mean_1kcal``, ``top5_delta_mean``, ``top10_delta_mean``.
    Z-score variants (``zscore_on_target``, ``zscore_off_target``,
    ``delta_z``, ``selectivity_rate_z``) are added when present in the input.
    No ``_1kcal`` Z-variants — Z is dimensionless, so a kcal cutoff doesn't apply.
    """
    scored = [r for r in rows if _as_float(r.get("delta")) is not None]
    n_valid = len(scored)
    n_total = len(rows)
    out: dict = {"n_valid": n_valid, "n_total": n_total}
    if n_valid == 0:
        return out

    deltas = _col(scored, "delta")
    on_scores = _col(scored, "score_on_target")
    sorted_deltas = sorted((d for d in deltas if d is not None))
    sel_1kcal = [d for d in deltas if d is not None and d <= -1]

    out.update({
        "score_on_target":        _mean(on_scores),
        "score_on_target_std":    _pstd(on_scores),
        "score_off_target":       _mean(_col(scored, "score_off_target")),
        "delta":                  _mean(deltas),
        "delta_std":              _pstd(deltas),
        "selectivity_rate":       _rate(deltas, 0),  # ← delta <= 0, matches legacy
        "selectivity_rate_1kcal": _rate(deltas, -1),
        "delta_mean_1kcal":       _mean(sel_1kcal),
        "delta_mean_1kcal_std":   _pstd(sel_1kcal),
        "top5_delta_mean":        _mean(sorted_deltas[:5]) if sorted_deltas else None,
        "top10_delta_mean":       _mean(sorted_deltas[:10]) if sorted_deltas else None,
    })

    # Z-score variants if present.
    if any("delta_z" in r for r in scored):
        z_scored = [r for r in scored if _as_float(r.get("delta_z")) is not None]
        if z_scored:
            dz = _col(z_scored, "delta_z")
            z_on = _col(z_scored, "zscore_on_target")
            out.update({
                "zscore_on_target":     _mean(z_on),
                "zscore_on_target_std": _pstd(z_on),
                "zscore_off_target":    _mean(_col(z_scored, "zscore_off_target")),
                "delta_z":              _mean(dz),
                "delta_z_std":          _pstd(dz),
                "selectivity_rate_z":   _rate(dz, 0),
                # τ=1 cutoff in z-space: Δ_z ≤ -1 i.e. on-target ≥ 1σ better than
                # off-target in the off-target's own reference distribution.
                "selectivity_rate_z_1": _rate(dz, -1),
            })
    return out


# ── safety-panel metrics ─────────────────────────────────────────────────────

def safety_metrics(rows: list[dict], target_names: list[str] | None = None) -> dict:
    """Given per-mol safety rows, return a flat dict of aggregate metrics.

    Per-target columns: ``score_<t>``, ``delta_<t>``, ``selectivity_rate_<t>``,
    ``selectivity_rate_1kcal_<t>``.

    Aggregate columns (see the module docstring for the "pool-then-rate"
    vs "rate-then-pool" distinction):

      ``score_on_target``, ``mean_safety_score``,
      ``delta_mean``, ``delta_mean_1kcal``,
      ``selectivity_rate_mean``, ``selectivity_rate_1kcal_mean``  (pool then rate),
      ``avg_selectivity_rate``, ``avg_selectivity_rate_1kcal``    (rate then mean),
      ``min_selectivity_rate``, ``min_selectivity_rate_1kcal``    (rate then min),
      ``worst_delta_mean``,
      ``strict_selectivity_rate``, ``strict_selectivity_rate_1kcal``  (rate on worst-per-mol),
      ``ratio20``.

    When ``target_names`` is explicitly given, ``mean_safety_score``,
    ``delta_mean``, and ``ratio20`` are recomputed from the individual
    per-target columns for that subset, rather than read from the
    pre-computed aggregate columns (which reflect all targets in the
    original evaluation run).

    Z-score variants added when present.
    """
    _recompute_aggregates = target_names is not None
    target_names = target_names or SAFETY_TARGETS
    scored = [r for r in rows if _as_float(r.get("mean_safety_score")) is not None]
    n_valid = len(scored)
    n_total = len(rows)
    out: dict = {"n_valid": n_valid, "n_total": n_total}
    if n_valid == 0:
        return out

    on_scores = _col(scored, "score_on_target")
    out["score_on_target"]     = _mean(on_scores)
    out["score_on_target_std"] = _pstd(on_scores)

    # per-target
    per_target_sr = []
    per_target_sr_1kcal = []
    for t in target_names:
        dv = _col(scored, f"delta_{t}")
        out[f"score_{t}"]                    = _mean(_col(scored, f"score_{t}"))
        out[f"delta_{t}"]                    = _mean(dv)
        sr   = _rate(dv, 0)
        sr1k = _rate(dv, -1)
        out[f"selectivity_rate_{t}"]         = sr
        out[f"selectivity_rate_1kcal_{t}"]   = sr1k
        if sr   is not None: per_target_sr.append(sr)
        if sr1k is not None: per_target_sr_1kcal.append(sr1k)

    # mean_safety_score, delta_mean, ratio20: recompute from target_names columns
    # when an explicit subset is requested; otherwise use the pre-computed columns
    # that reflect all targets from the original evaluation run.
    if _recompute_aggregates:
        per_mol_mean_scores, delta_means, per_mol_top20 = [], [], []
        for r in scored:
            s_on = _as_float(r.get("score_on_target"))
            t_scores = [v for t in target_names
                        if (v := _as_float(r.get(f"score_{t}"))) is not None]
            t_deltas = [v for t in target_names
                        if (v := _as_float(r.get(f"delta_{t}"))) is not None]
            if t_scores:
                per_mol_mean_scores.append(sum(t_scores) / len(t_scores))
            if t_deltas:
                delta_means.append(sum(t_deltas) / len(t_deltas))
            if s_on is not None and t_scores:
                all_s = sorted(t_scores + [s_on])
                per_mol_top20.append(int(s_on <= all_s[top20_threshold_idx(len(all_s))]))
        out["mean_safety_score"] = _mean(per_mol_mean_scores)
        out["ratio20"]           = _mean(per_mol_top20) if per_mol_top20 else None
    else:
        out["mean_safety_score"] = _mean(_col(scored, "mean_safety_score"))
        delta_means              = _col(scored, "delta_mean")
        out["ratio20"]           = _mean(_col(scored, "is_top20"))

    out["delta_mean"]                  = _mean(delta_means)
    out["delta_mean_std"]              = _pstd(delta_means)
    out["selectivity_rate_mean"]       = _rate(delta_means, 0)
    out["selectivity_rate_1kcal_mean"] = _rate(delta_means, -1)

    worst_deltas = []
    for r in scored:
        per = [_as_float(r.get(f"delta_{t}")) for t in target_names]
        per = [v for v in per if v is not None]
        if per:
            worst_deltas.append(max(per))  # least-negative = worst for selectivity
    out["worst_delta_mean"]              = _mean(worst_deltas)
    out["worst_delta_mean_std"]          = _pstd(worst_deltas)
    out["strict_selectivity_rate"]       = _rate(worst_deltas, 0)
    out["strict_selectivity_rate_1kcal"] = _rate(worst_deltas, -1)

    # avg/min across per-target rates
    out["avg_selectivity_rate"]       = _mean(per_target_sr)
    out["min_selectivity_rate"]       = min(per_target_sr) if per_target_sr else None
    out["avg_selectivity_rate_1kcal"] = _mean(per_target_sr_1kcal)
    out["min_selectivity_rate_1kcal"] = min(per_target_sr_1kcal) if per_target_sr_1kcal else None

    # z-score variants. No "_1kcal" threshold — Z-scores are dimensionless.
    if any("delta_z_mean" in r for r in scored):
        z_scored = [r for r in scored if _as_float(r.get("delta_z_mean")) is not None]
        if z_scored:
            z_on = _col(z_scored, "zscore_on_target")
            if _recompute_aggregates:
                # Recompute mean_zscore_safety, delta_z_mean, ratio20_z from
                # per-target zscore_{t} columns for the requested target subset.
                per_mol_mzs, per_mol_dz, per_mol_top20_z = [], [], []
                for r in z_scored:
                    z_on_r = _as_float(r.get("zscore_on_target"))
                    zt_vals = [v for t in target_names
                               if (v := _as_float(r.get(f"zscore_{t}"))) is not None]
                    if z_on_r is None or not zt_vals:
                        continue
                    mzs = sum(zt_vals) / len(zt_vals)
                    per_mol_mzs.append(mzs)
                    per_mol_dz.append(z_on_r - mzs)
                    all_z = sorted([z_on_r] + zt_vals)
                    per_mol_top20_z.append(int(z_on_r <= all_z[top20_threshold_idx(len(all_z))]))
                dz = per_mol_dz
                out.update({
                    "zscore_on_target":     _mean(z_on),
                    "zscore_on_target_std": _pstd(z_on),
                    "mean_zscore_safety":   _mean(per_mol_mzs),
                    "delta_z_mean":         _mean(dz),
                    "delta_z_mean_std":     _pstd(dz),
                    "ratio20_z":            _mean(per_mol_top20_z),
                })
            else:
                dz = _col(z_scored, "delta_z_mean")
                out.update({
                    "zscore_on_target":     _mean(z_on),
                    "zscore_on_target_std": _pstd(z_on),
                    "mean_zscore_safety":   _mean(_col(z_scored, "mean_zscore_safety")),
                    "delta_z_mean":         _mean(dz),
                    "delta_z_mean_std":     _pstd(dz),
                    "ratio20_z":            _mean(_col(z_scored, "is_top20_z")),
                })
            # Per-target z-SRs (threshold-then-mean), so we can report a true
            # z-analogue of avg/min selectivity_rate. Per-mol z-Δ for target t
            # is computed inline as zscore_on_target − zscore_<t> (same on−off
            # convention as raw delta_<t>; more-negative = more-selective).
            # τ=0 → "any selectivity"; τ=1 → on-target ≥1σ better in target's
            # reference distribution (z-space stand-in for the raw 1-kcal cut).
            per_target_sr_z = []
            per_target_sr_z_1 = []
            for t in target_names:
                dzt = []
                for r in z_scored:
                    zo = _as_float(r.get("zscore_on_target"))
                    zt = _as_float(r.get(f"zscore_{t}"))
                    if zo is not None and zt is not None:
                        dzt.append(zo - zt)
                sr_z = _rate(dzt, 0)
                sr_z_1 = _rate(dzt, -1)
                if sr_z is not None:
                    per_target_sr_z.append(sr_z)
                if sr_z_1 is not None:
                    per_target_sr_z_1.append(sr_z_1)
                out[f"zscore_{t}"] = _mean(_col(z_scored, f"zscore_{t}"))
            out["avg_selectivity_rate_z"]   = _mean(per_target_sr_z)
            out["min_selectivity_rate_z"]   = (min(per_target_sr_z)
                                               if per_target_sr_z else None)
            out["avg_selectivity_rate_z_1"] = _mean(per_target_sr_z_1)
            out["min_selectivity_rate_z_1"] = (min(per_target_sr_z_1)
                                               if per_target_sr_z_1 else None)

    return out


# ── IO helpers ───────────────────────────────────────────────────────────────

_SCORE_SANITY_BOUND = 100.0  # |kcal/mol| ceiling; Vina is bounded ~[-20, +20]


def load_raw_rows(csv_path: str) -> list[dict]:
    """Read a per-mol CSV, dropping any legacy ``summary`` row if present.

    Also drops rows with absurd Vina scores (|score| > 100 kcal/mol) — these
    indicate a failed docking that returned uninitialized memory or a
    malformed receptor/ligand pair. Bug observed for speclig mol_idx=20
    (sulfur-H valence) producing scores around 1.25e8.
    """
    import csv as _csv
    rows = []
    n_dropped = 0
    with open(csv_path, newline="") as f:
        for r in _csv.DictReader(f):
            if r.get("mol_idx") == "summary":
                continue
            try:
                s_on = r.get("score_on_target")
                s_off = r.get("score_off_target")
                if s_on is not None and s_on != "" and abs(float(s_on)) > _SCORE_SANITY_BOUND:
                    n_dropped += 1; continue
                if s_off is not None and s_off != "" and abs(float(s_off)) > _SCORE_SANITY_BOUND:
                    n_dropped += 1; continue
            except (ValueError, TypeError):
                pass  # let downstream handle non-numeric
            # safety eval: any score_<target> field can also be polluted
            for k, v in r.items():
                if k.startswith("score_") and k not in ("score_on_target", "score_off_target"):
                    try:
                        if v is not None and v != "" and abs(float(v)) > _SCORE_SANITY_BOUND:
                            n_dropped += 1
                            break
                    except (ValueError, TypeError):
                        continue
            else:
                rows.append(r)
    if n_dropped:
        print(f"  [load_raw_rows] dropped {n_dropped} sanity-fail rows from "
              f"{os.path.basename(csv_path)} (|score| > {_SCORE_SANITY_BOUND})")
    return rows
