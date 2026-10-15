# scripts/summarize/

Cross-method summarisation tools:

```
summarize/
├── _common.py              # Shared helpers; path constants, ref-stats loader, z-score augmentation, cap-to-n-valid, write_csv.
├── cross_method.py         # reduces the raw per-mol CSVs into one row per method.
└── summarize_boltz2.py     # per-case cross-model summary for Boltz2 affinity runs
```

## File flow

```
results/
├── <method>/                                            ← raw per-mol CSVs (read by cross_method.py)
│   ├── cs/<case>.csv                                    # parp.csv, drd2_hrh1_jmed.csv, ...
│   ├── safety/<poi>.csv                                 # parp1.csv, jak2.csv, ...
│   └── crossdocked/<on>_vs_<off>__<diff>.csv            # 2pqw_A_vs_2rhy_A__hard.csv, ...
│
├── <method>_compose/                                    ← selectivity-guided runs; see "Compose methods" below
│
└── summary/                                             ← written by cross_method.py
    ├── cs/
    │   └── <case>.csv            ← rows = methods, one CSV per case
    │
    ├── safety/
    │   ├── <poi>.csv             ← rows = methods, full panel (84 anti-targets)
    │   └── <poi>_priority.csv    ← rows = methods, priority subset (22)
    │
    └── crossdocked/
        └── <diff>.csv          ← rows = methods, pair-weighted within tier
```

Compose runs live in `<method>_compose/` dirs and are handled separately — see
[Compose methods](#optional-compose-methods-selectivity-guided--provided-for-the-paper)
at the end. If you're summarising your own method you can ignore them.

---

## `cross_method.py`
> **Note:** `cross_method.py` only summarizes CSVs **under the `results/<method>/{cs,safety,crossdocked}/` layout** that `run_eval.py` writes.
> A direct `selectbench evaluate-*` run with a custom `--output` (or its default
> `results/evaluate-*/…` path) will **not** be picked up by the summarizer
> unless you place it under that layout. For the generate→evaluate→summarize
> flow, drive evaluation with `run_eval.py`.

Reads raw per-mol CSVs and writes one summary CSV per (case | poi | crossdocked target), with one row per method.

```bash
python scripts/summarize/cross_method.py                      # default --mode all = cs + safety + crossdocked
python scripts/summarize/cross_method.py --mode cs            
python scripts/summarize/cross_method.py --mode safety
python scripts/summarize/cross_method.py --mode crossdocked
```

### Auto-discovery

When `--methods` is omitted, every `results/<name>/` subdir that contains at least one of `cs/`, `safety/`, or `crossdocked/` is included. 
Drop a new method's results into `results/<new_method>/{cs,safety,crossdocked}/...` and re-run — no script edit required.

### Outputs

| `--mode` | Output path | Shape |
|---|---|---|
| `cs` | `results/summary/cs/<case>.csv` | rows = method, cols = CS metrics |
| `safety` | `results/summary/safety/<poi>.csv` (full panel, 84 anti-targets) **and** `results/summary/safety/<poi>_priority.csv` (priority subset, 22) | rows = method, cols = safety metrics + per-target Δ/SR |
| `crossdocked` | `results/summary/crossdocked/<diff>.csv` | rows = method, pair-weighted within tier |

Safety output is written as a pair (full + priority). A raw CSV only feeds the panels it was docked against: a run made with `evaluate-safety --priority_targets` appears in the priority table only. Save such a run as `<poi>.csv` like any other — the `<poi>_priority` name is reserved for the summary and is skipped if used for raw results. `htr1b_agonist` is excluded from both lists at the metrics-module level 
(see [`data/README.md`](../../data/README.md#safety_targetssafety_targetscsv--85-anti-target-entries-84-reported-after-htr1b_agonist-exclusion) for the rationale). Aggregate columns like `mean_safety_score`, `delta_mean`, and `ratio20` are **always** recomputed from the per-target columns (the summarizer passes an explicit target list on every call), so panel numbers stay self-consistent with the scope being reported. Note the consequence: if the raw eval was run with `--targets` over a narrower set, the `mean_safety_score` stored in that CSV is recomputed rather than reused.

### Z-score columns

Z-score columns are added in-memory from the reference-stats CSVs at:

| File | Used for |
|---|---|
| `data/zscore_library/processed/case_studies_stats.csv` | CS pockets |
| `data/zscore_library/processed/standard_docking_benchmark_stats.csv` + `safety_targets_stats.csv` | safety panel (loaded together) |
| `data/zscore_library/processed/crossdocked_test_set_stats.csv` | CrossDocked pockets |

Pass empty strings to `--cs_reference_stats` / `--safety_reference_stats` /
`--crossdocked_reference_stats` to skip a given augmentation.

### All flags

| Argument | Default | Description |
|---|---|---|
| `--score_fn` | `docking` | Scoring backend: `docking` (default) reads Vina docking CSVs; `boltz2` reads Boltz2 affinity CSVs instead |
| `--mode` | `all` | `cs`, `safety`, `crossdocked`, or `all` (= cs + safety + crossdocked) |
| `--methods` | auto-discover | Subset of method dirs to include |
| `--out_dir` | `results/summary` | Output base dir. Under `--score_fn boltz2` the default becomes `results/summary/boltz2` |
| `--results_dir` | `results` | Input base dir |
| `--cs_reference_stats` | (default file) | Stats CSV for CS z-scores; `""` disables |
| `--safety_reference_stats` | (default pair) | Pathsep-joined stats CSVs for safety z-scores; `""` disables |
| `--crossdocked_reference_stats` | (default file) | Stats CSV for crossdocked z-scores; `""` disables |
| `--cap_n_valid` | `100` | Truncate each method to the first this-many valid molecules per file before computing metrics, so all methods are compared on an equal molecule budget (`0` disables) |
| `--case_ids` | none | (`crossdocked` only) restrict to these case_ids |
| `--sweep_detail` | off | Also emit `<name>_sweeps.csv` for cs/safety with every compose sweep point as its own `<method>_<sweep>` row (best-only files still written) |
| `--quality` | off | Append reference-free molecular-quality columns (QED, SA, LogP, Lipinski, MW, Diversity) computed from each raw CSV's `smiles`. Off by default, so selectivity-only output is unchanged. Requires RDKit; docking `--score_fn` only. See [`results/README.md`](../../results/README.md#metric-glossary-summary). |

> **If one wants to add new/custom metrics**: metrics are computed on the fly by `scripts/summarize/cross_method.py`, so adding a new metric only requires editing 
> `src/selectbench/metrics.py`. Z-score columns materialise in the *summary* CSVs (`results/summary/...`) when `cross_method.py` is invoked with the relevant
> `--<kind>_reference_stats` file, which augments rows in memory before aggregation.

## Boltz2 as an alternative score function

`cross_method.py` also supports Boltz2 affinity-probability inputs via `--score_fn boltz2`. Instead of Vina scores it reads per-mol probabilities from `results/boltz2/<model>/cs/<case>.csv` (written by `scripts/run_boltz2_case_study.py`) and emits one summary row per model:

```bash
python scripts/summarize/cross_method.py --score_fn boltz2
# → results/summary/boltz2/cs/<case>.csv
```

For a per-case cross-model table (one CSV per case, one row per model, with mean ± std of `on_target_affinity_prob` and delta), use `summarize_boltz2.py`.

By default it only prints the table and, with `--output`, writes it as CSV. Three
further outputs are **opt-in** — pass the flag to enable each:

| Flag | Writes |
|---|---|
| `--output PATH` | the summary table as CSV |
| `--selective_output_dir DIR` | per-`(case, model)` CSVs of molecules ranking high on-target and low off-target (on-target rank > 0.6, off-target rank < 0.4) |
| `--top_delta_output_dir DIR` | per-`(case, model)` CSVs of the top-10 molecules by affinity delta |
| `--latex_output PATH` | a LaTeX `longtable` of the summary. Note this uses SelectBench paper-specific macros (`\mycaption`, `\datasetA`) and will not compile outside the paper source |
| `--cap N` | cap each `(model, case)` at N rows (default 100; `0` disables) |
| `--results_dir DIR` | input base dir for Boltz2 CSVs (default `results/boltz2`) |

See [`BOLTZ2.md`](../../BOLTZ2.md) for install and usage.

## [Optional] Compose methods (selectivity-guided — provided for the paper)

> `compose` is the selectivity-guided generation method we benchmark in the paper (produced by the `scripts/generate_compose/` wrappers). 
> It is a separate method, not part of the core eval loop: **if you're summarising your own method you can ignore this section** 
> — the auto-discovery above already handles a plain `results/<method>/` dir. This section is only relevant when you have `results/<method>_compose/` dirs to aggregate.

A compose run differs from a normal method only in that a single dir holds many guidance-strength sweep points and two selection modes, encoded in each filename:

```
results/<method>_compose/
├── cs/<case>_bon<x>_boff<y>_<MODE>.csv
├── safety/<poi>_bon<x>_boff<y>_<MODE>.csv                  # K=1 (single off-target)
├── safety/<poi>_<K>off_<scope>_bon<x>_boff<y>_<MODE>.csv   # K=2, 3 (multi-off; scope ∈ priority|all)
└── crossdocked/<on>_vs_<off>__<diff>_bon<x>_boff<y>_<MODE>.csv
```

- `<MODE>` ∈ `{CFG-SDE, FKC}` — the selection mode.
- `bon<x>_boff<y>` — the guidance-strength sweep point.
- `<scope>` ∈ `{priority, all}` — which hardest-target ranking a multi-off run
  steered against (compose wrapper's `--hardest_scope`, default `priority`).

**Auto-split into per-mode synthetic methods.** Any results dir whose name ends
in `_compose` is split by `cross_method.py` into one summary row per mode (and
per K), so each becomes its own comparable "method":

- K=1 (single off-target): `diffsbdd_compose` → `diffsbdd_compose_cfgsde`, `diffsbdd_compose_fkc`.
- K=2, 3 (multi off-target): also `diffsbdd_compose_2off_cfgsde`,
  `diffsbdd_compose_2off_fkc`, `diffsbdd_compose_3off_cfgsde`,
  `diffsbdd_compose_3off_fkc` — one per (K, mode) — when the dir contains files
  with the `<K>off_<scope>` infix.

**Best sweep point wins.** When multiple sweep points exist for the same
(method, target, K, mode), the file with the best mean `delta` is kept (raw sign
convention: `score_on − score_off`, so smaller = more selective). Pass
`--sweep_detail` to **keep every sweep point** instead: alongside each `cs`/`safety`
summary it writes a parallel `<name>_sweeps.csv` (`<case>_sweeps.csv`,
`<poi>_sweeps.csv`, `<poi>_priority_sweeps.csv`) with one row per sweep point,
labelled `<method>_<sweep>` (e.g. `diffsbdd_compose_fkc_bon2.5_boff-0.5`).
Crossdocked tables are pair-aggregated (best-by-delta per pair) and get no
`_sweeps.csv`.

**This filename-based pickup applies only inside a `_compose` dir.** 
A dir whose name ends in `_compose` is *not* read by ordinary auto-discovery; instead each file in it is matched by filename, and a file is aggregated only when both hold:

1. its `<MODE>` is `CFG-SDE` or `FKC` (`COMPOSE_MODES`); **and**
2. it carries a sweep token (`bon<x>_boff<y>`, or the default-guidance `w<w>` form).

A file in a `_compose` dir that matches neither is silently skipped at summary time — it can still be generated and docked, it just isn't aggregated.