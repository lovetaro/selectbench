# Results Directory
> This directory documents the layout and column schema. The whole `results/` tree is git-ignored and regenerated on demand by **evaluate → summarize** flow. 
> You can build a small filled-in reference to anchor your own output against: drop a handful of SMILES into `generated_samples/<method>/cs/<case>.smi`, then run

```bash
python scripts/run_eval.py --method <method> --eval cs --exhaustiveness 8
#   → results/<method>/cs/<case>.csv          (raw per-mol)
python scripts/summarize/cross_method.py --mode cs --methods <method>
#   → results/summary/cs/<case>.csv           (cross-method summary)
```

## Schema (full pipeline output):

```
results/
  <method>/                                 # "raw per-mol" csv
    cs/<case>.csv                           # one per case study
    safety/<poi>.csv                        # one per safety PoI
    crossdocked/<on>_vs_<off>__<diff>.csv   # one per oriented CrossDocked pair
  summary/                                  # cross-method "aggregates"
    cs/<case>.csv                           # rows = method, cols = metrics
    safety/<poi>.csv                        # full panel (84 anti-targets)
    safety/<poi>_priority.csv               # priority subset (22 anti-targets)
    crossdocked/<diff>.csv                  # pair-weighted within tier, rows = method
```
For compose methods we keep the same layout but encode sweep params and selection mode in the filename: `<base>_bon<x>_boff<y>_<MODE>.csv`. 

## Metric glossary (summary/)
> **Sign convention — read this once.** Raw per-mol columns follow the legacy Vina convention `delta = score_on − score_off` (lower = stronger on-target binding, so `delta ≤ 0` ⇒ *selective*). 
> Summary CSVs flip this for readability AND rename the flipped columns to `delta_selectivity_*` (higher = more selective), 
> so the two conventions **never share a column name**. 
> If you see `delta_*` in a file, it's the raw legacy sign; if you see `delta_selectivity_*`, it's the flipped summary sign. 
> SR-style fractions and `_std` columns are sign-invariant and stay under their original names.

The raw per-mol CSVs hold scores; the numbers you actually *read and compare across methods* are the aggregates in `results/summary/…`, 
computed by `cross_method.py`. This is what each one means. 
**Unless noted, every rate is a fraction in `[0, 1]` and higher = more selective, and every `delta_selectivity*` is in kcal/mol with higher = more selective** 
(it is the sign-flipped `off − on` margin — see the sign convention above). Raw Vina `score_*` columns are the exception: they are kcal/mol where 
*lower = stronger binding*.

### Case study & CrossDocked (`summary/cs/`, `summary/crossdocked/`)

| Metric | Plain-language meaning | Range / good direction |
|---|---|---|
| `delta_selectivity` / `delta_selectivity_std` | **the headline number**: mean selectivity margin over all valid molecules | kcal/mol, higher better |
| `selectivity_rate` | fraction of molecules that bind the on-target **at least as well as** the off-target (`P(delta ≤ 0)`) | `[0,1]`, higher better |
| `selectivity_rate_1kcal` | stricter: fraction that bind the on-target **≥ 1 kcal/mol stronger** than the off-target (`P(delta ≤ −1)`) | `[0,1]`, higher better |
| `delta_selectivity_mean_1kcal` | mean selectivity margin (kcal/mol) among the molecules that clear the 1-kcal bar | kcal/mol, higher better |
| `top5_delta_selectivity_mean` / `top10_delta_selectivity_mean` | best-case: mean margin of the 5 / 10 most selective molecules | kcal/mol, higher better |
| `selectivity_rate_z`, `delta_selectivity_z`, `zscore_on_target` | z-score analogues, normalized against each target's reference-ligand distribution (dimensionless; comparable across targets). `_z_1` = a 1σ cutoff, the dimensionless analogue of the 1-kcal cut | higher better |

### Safety panel (`summary/safety/`)

The panel scores one on-target against ~84 anti-targets, so there are several ways to pool "selective vs the panel" — they measure genuinely different things:

| Metric | Plain-language meaning | Range / good direction |
|---|---|---|
| `selectivity_rate_mean` | **pool-then-rate**: for each molecule average its margin across the panel, then take the rate — "selective *on average* across the panel" | `[0,1]`, higher better |
| `avg_selectivity_rate` | **rate-then-mean**: selectivity rate per anti-target, then averaged over targets | `[0,1]`, higher better |
| `min_selectivity_rate` | the **worst** per-target selectivity rate — the single anti-target the method is *least* selective against | `[0,1]`, higher better |
| `strict_selectivity_rate` | hardest bar: fraction of molecules that beat **every** anti-target simultaneously (per-mol worst margin ≤ 0) | `[0,1]`, higher better |
| `worst_delta_selectivity_mean` | mean over molecules of each molecule's *least-selective* margin | kcal/mol, higher better |
| `ratio20` | fraction of molecules whose on-target score lands in the top-20 % across the on-target + full panel | `[0,1]`, higher better |
| `*_1kcal` suffix | the same metric with a 1-kcal/mol margin required instead of ≥ 0 | stricter variant |
| `*_z`, `*_z_1` suffix | z-space analogues (dimensionless, reference-normalized); `_1` = 1σ cutoff | higher better |

Also emitted but not tabulated above: `score_on_target_std`,
`score_off_target`, `zscore_off_target`, `delta_selectivity_z_std`, and — in
`summary/crossdocked/` only — `n_pairs` (how many oriented pairs have at least
one scored molecule).
The safety tables additionally carry `mean_safety_score`,
`mean_zscore_safety`, and a per-target block of
`delta_selectivity_<t>` / `selectivity_rate_<t>` / `selectivity_rate_1kcal_<t>`
for every panel target.

Every summary row starts with `n_valid` / `n_total`: molecules with a usable
score, and molecules read. In `summary/crossdocked/` they are totals across the
tier's pairs.

`summary/crossdocked/` averages each metric over the tier's pairs. A metric
that is undefined for some pairs is averaged over the pairs where it is defined,
and gets a companion `<metric>_n_pairs` column giving that count. In practice
this is `delta_selectivity_mean_1kcal` (and its `_std`), which has no value for
a pair in which no molecule clears 1 kcal/mol — read it alongside
`delta_selectivity_mean_1kcal_n_pairs`, not `n_pairs`.

Every metric above is defined in code in `src/selectbench/metrics.py`
(`cs_metrics`, `safety_metrics`) — but under its **raw** name; grepping the
`delta_selectivity_*` names above in `metrics.py` finds nothing.
`scripts/summarize/cross_method.py` applies the sign flip and the rename
(see `_DELTA_RENAME_MAP` and `_apply_sign_convention`). This table is the
plain-language mirror of the emitted summary columns.

### Molecular quality metrics (opt-in)

Alongside selectivity, SelectBench can report standard **reference-free** molecular-quality metrics — a pure function of the generated SMILES, independent of docking or on-/off-target pairing. The sign convention does **not** apply to these columns (they are not deltas). Definitions are in `src/selectbench/sbdd_metrics.py`, following DiffSBDD's `MoleculeProperties`:

| Metric | Plain-language meaning | Range / good direction |
|---|---|---|
| `qed_mean` | mean QED drug-likeness | `[0,1]`, higher better |
| `sa_mean` | mean synthetic accessibility, normalized `(10 − raw_SA)/9` | `[0,1]`, higher = easier to make |
| `logp_mean` | mean Crippen LogP (octanol–water partition) | descriptive (drug-like ≈ 1–3) |
| `lipinski_mean` | mean count (0–5) of Lipinski-style rules satisfied (MW<500, HBD≤5, HBA≤10, −2≤logP≤5, RotBonds≤10) | `[0,5]`, higher better |
| `mw_mean` | mean molecular weight (Da) | descriptive |
| `diversity` | mean pairwise Tanimoto **distance** (`1 − similarity`, RDKit fingerprints) over all valid molecules (no de-duplication, so mode collapse lowers the score), capped at 1000 | `[0,1]`, higher = more diverse |
| `n_scored` | molecules with a computable property (invalid SMILES excluded) | count |

Each `*_mean` has a matching `*_std` (population stdev). In the CrossDocked summary these columns are pair-averaged within a difficulty tier, like every other numeric column there.

## Raw per-mol csv
> **Z-scores are a summary-time option, not a raw-CSV column.** 
> If a reference-stats CSV is supplied via `--safety_reference_stats` to `cross_method.py`, per-molecule
> z-scores (of `score_on_target` and each `score_<safety_target>` against that
> target's reference-ligand distribution) are computed **internally** and surfaced
> only as the aggregate `*_z` columns below — they are never written to the raw per-mol CSV.

### CS files (`cs/<case>.csv`)

One row per generated molecule; one CSV per case study.

| Column | Description |
|--------|-------------|
| `mol_idx` | Index of the molecule in the generated set |
| `smiles` | SMILES of the generated molecule |
| `score_on_target` | Vina docking score (kcal/mol) on the intended on-target receptor (lower = stronger binding). Empty = docking failed. |
| `score_off_target` | Vina docking score on the off-target receptor. Empty = docking failed. |
| `delta` | `score_on_target − score_off_target` (per-mol, **raw legacy sign**: negative ⇒ selective for the on-target). Summary CSVs rename the flipped version to `delta_selectivity` (positive ⇒ selective). |

Aggregates are computed by `scripts/summarize/cross_method.py` and emitted into `results/summary/cs/<case>.csv`. Some columns are conditional: `delta_selectivity_mean_1kcal` is only emitted when at least one molecule clears the 1-kcal bar (i.e. some per-mol `delta ≤ −1`);  
Z-score columns require a reference-stats CSV via `--cs_reference_stats`.

### CrossDocked files (`crossdocked/<on>_vs_<off>__<diff>.csv`)
Same per-mol shape as CS files, with one CSV per oriented `(on, off)` pocket pair from `data/crossdocked_test_set/crossdocked_test_set_pairs.csv`. Columns are identical to CS files. Aggregates are calculated per each difficulty tier and emitted by `cross_method.py` into `results/summary/crossdocked/<diff>.csv`. 

### Safety files (`safety/<poi>.csv`)
One row per generated molecule; one CSV per Protein-of-Interest (PoI).

| Column | Description |
|--------|-------------|
| `mol_idx` | Index of the molecule |
| `smiles` | SMILES |
| `score_on_target` | Vina score on the PoI receptor (kcal/mol; empty = docking failed) |
| `score_<safety_target>` | Vina score on each safety panel target (e.g. `score_hrh1_antagonist`, `score_ache_inhibitor`). One column per panel target. Empty = docking failed for that target. |
| `mean_safety_score` | Mean Vina score across the panel (excludes failed targets — see `n_safety_scored`) |
| `n_safety_scored` | How many panel targets actually produced a score for this molecule. |
| `n_safety_total` | Panel size for the run. When this exceeds `n_safety_scored`, every panel-derived column for that molecule (`mean_safety_score`, `delta_mean`, `is_top20`) rests on the subset that scored, which makes the molecule look **more** selective than it is — `strict_selectivity_rate` takes a max over only the targets present, and a max over a subset is never larger. A gap here is a docking failure, usually memory pressure killing the Vina subprocess, not evidence the molecule avoids those targets: a molecule that genuinely cannot bind a target still gets a (bad) number, not an empty cell. |
| `delta_<safety_target>` | `score_on_target − score_<safety_target>` (per-mol, **raw legacy sign**: negative ⇒ selective). Summary CSVs rename the flipped version to `delta_selectivity_<t>`. |
| `delta_mean` | `score_on_target − mean_safety_score` (per-mol, raw legacy sign). Summary rename: `delta_selectivity_mean`. |
| `is_top20` | `1` iff `score_on_target` is in the top-20 % across the on-target plus all panel scores for this molecule, else `0`. Empty = the on-target docking failed. |

Failure reasons are printed to stdout only; the CSV records a failure as an
empty score cell. Capture the run's console output if you need to know *why* a
cell is blank.