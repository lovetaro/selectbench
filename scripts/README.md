# scripts/

Utility scripts for generating, evaluating, and summarising SelectBench benchmarks. 

> **The generation wrappers assume the baseline is already installed.**
> `generate_compose/` (plus the DecompDiff unguided wrapper at
> `baselines/decompdiff/gen_decompdiff.py`) are the *example* wrappers we used
> to produce the paper's numbers — they shell out to each baseline's own upstream
> code (DiffSBDD, DecompDiff, …). SelectBench does **not** vendor every baseline,
> so clone/install the one you want first (see [`baselines/README.md`](../baselines/README.md)) 
> and point the wrapper at its checkpoint.
>
> **Already have molecules from your own method?** Skip the generation wrappers
> entirely: drop your SDF/SMILEs into `generated_samples/<yourmethod>/<benchmark_type>/` and start
> at the **evaluate** step in [End-to-end workflows](#end-to-end-workflows) —
> `run_eval.py` docks them and `summarize/cross_method.py` produces the
> cross-method metrics table.

```
scripts/
├── generate_compose/            # selectivity-guided wrappers      — see generate_compose/README.md
├── summarize/                   # cross-method tables              — see summarize/README.md
├── run_eval.py                  # evaluate dispatcher (see below)
├── hardest_safety_targets.py    # per-(method, PoI) safety-target difficulty ranking
└── run_boltz2_case_study.py     # Boltz2 affinity batch runner: on/off targets of a case study (see ../BOLTZ2.md)
```

**Per-subdirectory docs:** [`generate_compose/`](generate_compose/README.md) · [`summarize/`](summarize/README.md) · [`Boltz2`](../BOLTZ2.md)

---

## Directory layout

Generation, evaluation, and summarisation share a parallel three-bucket
layout (`cs/`, `safety/`, `crossdocked/`) so the same script can fan out
across benchmark types without colliding filenames.

```
generated_samples/
  <method>/
    cs/<case>.sdf                                    # parp.sdf, ...
    safety/<poi>.sdf                                 # parp1.sdf, ...
    crossdocked/<on>.sdf                             # e.g. 3tym_A.sdf, ...
  <guided_method>/
    cs/<case>_bon<x>_boff<y>_<MODE>.sdf
    safety/<poi>_bon<x>_boff<y>_<MODE>.sdf
    crossdocked/<on>_vs_<off>__<diff>_bon<x>_boff<y>_<MODE>.sdf

results/
  <method>/
    cs/<case>.csv
    safety/<poi>.csv
    safety/hardest_safety_targets.csv
    crossdocked/<on>_vs_<off>__<diff>.csv
  <guided_method>/
    cs/<case>_bon<x>_boff<y>_<MODE>.csv
    safety/<poi>_bon<x>_boff<y>_<MODE>.csv
    crossdocked/<on>_vs_<off>__<diff>_bon<x>_boff<y>_<MODE>.csv
  summary/
    cs/<case>.csv                                    # rows = methods, one CSV per case
    safety/<poi>.csv                                 # rows = methods, full panel (84 anti-targets)
    safety/<poi>_priority.csv                        # rows = methods, priority subset (22)
    crossdocked/<diff>.csv                           # rows = methods, pair-weighted per tier
```

`<on>` / `<off>` are `<pdb>_<chain>` pocket identifiers; `<diff>` is the
CrossDocked difficulty tier (`easy`, `medium`, or `hard`); `<MODE>` is `CFG-SDE`, `FKC` for compose outputs.

---

## End-to-end workflows

Each benchmark type follows the same shape:
**generate → eval (raw docking scores) → cross_method (summarize, rows = methods).**

> **The `# 1. Generate` steps below use `gen_decompdiff.py` as a concrete
> example and assume you have the DecompDiff baseline cloned and set up** (see
> [`baselines/README.md`](../baselines/README.md)). Swap in the wrapper (or
> upstream command) for your baseline, or — if you already have generated SDFs —
> skip generation entirely and start at the **`# 2. Evaluate`** step.

### Refresh everything (after evals are populated)

```bash
python scripts/summarize/cross_method.py    # cs + safety + crossdocked
```

The per-benchmark snippets below show the granular form when you want to
refresh just one bucket.

### CS — case study

```bash
# 1. Generate
python baselines/decompdiff/gen_decompdiff.py --case_study <case> --device cuda:0
# (compose: see generate_compose/README.md)

# 2. Evaluate
python scripts/run_eval.py --eval cs        # auto-discovers all methods

# 3. Cross-method summary
python scripts/summarize/cross_method.py --mode cs
```

### Safety — multi-anti-target panel

```bash
# 1. Generate
python baselines/decompdiff/gen_decompdiff.py --poi <poi> --device cuda:0
# (compose: see generate_compose/README.md)

# 2. Evaluate
python scripts/run_eval.py --eval safety

# 3. Rank each PoI's anti-targets hardest→easiest from the eval scores. 
# Only needed if you'll run Safety compose in --poi auto mode: there the wrapper steers away from the top-N hardest off-targets read from this ranking, instead of you naming one with --safety_target.
# Writes results/<method>/safety/hardest_safety_targets.csv.
python scripts/hardest_safety_targets.py

# 4. Cross-method summary
python scripts/summarize/cross_method.py --mode safety
```

### CrossDocked (sb-xl)

```bash
# 1. Generate. 
# Pick CrossDocked on-targets / pairs by case_id OR a difficulty tier 
# (see data/README.md "Enumerate targets" and generate_compose/README.md "Choosing a CrossDocked pair"), 
# then run the per-target wrappers. Baseline (unguided) on-target:
python baselines/decompdiff/gen_decompdiff.py --crossdocked <pdb>_<chain> --device cuda:0

# Compose pair (one SDF per oriented (on,off) pair, selected by case_id).
# Run from the repo root — the wrapper cds into the DiffSBDD repo internally and resolves the relative --checkpoint against it (see generate_compose/README.md).
python scripts/generate_compose/gen_compose_diffsbdd.py --crossdocked_case 0 \
    --checkpoint baselines/diffsbdd/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt \
    --beta_on 2.5 --beta_off -0.5 --reverse_method SDE --resample --device cuda:0

# 2. Evaluate (baseline SDFs auto-fan across off pairs; compose maps 1-1 by stem)
python scripts/run_eval.py --eval crossdocked --nproc 32

# 3. Cross-method summary (one CSV per difficulty tier, pair-weighted within tier)
python scripts/summarize/cross_method.py --mode crossdocked
```

For per-flag references on each generation/summarize wrapper, see the subdirectory README (linked at the top).

---

## `run_eval.py`

Auto-discovers SDFs / SMIs in `generated_samples/<method>/<benchmark_type>/` and runs
the corresponding `selectbench evaluate*` CLI command. Output goes to
`results/<method>/<benchmark_type>/<name>.csv`. This is the batch entry point; to score a
single file to an explicit path, call the `selectbench evaluate-*` subcommand
directly with `--output` instead (both share the same docking core and
defaults).

```bash
# Auto-discover every method that has files in the requested <benchmark_type>/ bucket
python scripts/run_eval.py --eval cs
python scripts/run_eval.py --eval safety
python scripts/run_eval.py --eval crossdocked --nproc 32

# Pin to specific method(s)
python scripts/run_eval.py --method <method> <method>_compose --eval cs
python scripts/run_eval.py --method <method>_compose --eval safety

# Single file (requires exactly one --method so the results dir is unambiguous)
python scripts/run_eval.py --method <method> --eval cs \
    --sdf generated_samples/<method>/cs/<case>.sdf
```

| Argument | Default | Description |
|---|---|---|
| `--method` | auto-discover | Method name(s); omit to scan `generated_samples/*/` |
| `--eval` | `cs` | `cs` (alias `case`), `safety`, or `crossdocked` |
| `--sdf` | auto | Explicit SDF/SMI paths; skips auto-discovery (one method only) |
| `--case_study` / `--poi` | inferred | Override the inferred target name |
| `--targets` | | Specific safety target(s); repeatable |
| `--priority_targets` | | Use only the priority safety subset |
| `--nproc` | `min(8, $SLURM_CPUS_PER_TASK or cpu_count())` | Parallel docking workers |
| `--exhaustiveness` | per-eval (case=20, crossdocked=10, safety=1) | Vina exhaustiveness |
| `--timeout` | `600` | Per-molecule docking timeout (seconds); exported as `SELECTBENCH_DOCK_TIMEOUT` |
| `--out_csv` | auto | Override the output CSV filename (joined with the eval-type subdir) |
| `--case_ids` | | (crossdocked only) Comma-separated override of the auto fan-out |
| `--no_fallback` (alias `--no-fallback`) | off | Disable the low-exhaustiveness retry; forwarded as `--no-fallback` to each `selectbench` subcommand (see [Docking fallback](#docking-fallback)) |
| `--results_name` | same as `--method` | Override the results subdir name |
| `--results_dir` | `<repo>/results` | Base results directory; output lands in `<results_dir>/<method>/<bucket>/` |
| `--dry_run` | | Print commands without running |

### Docking fallback

Each underlying `selectbench evaluate*` subcommand has a fallback retry **on
by default** for any row whose primary score is `None` (Vina failure /
timeout / parse error). Failed rows are re-run **serially (`nproc=1`)** with
a lower exhaustiveness and a longer per-mol timeout:

| Eval mode | First-pass exhaustiveness | First-pass timeout | Fallback exhaustiveness | Fallback timeout |
|---|---|---|---|---|
| `case` | 20 | 600s | **8** | **1200s** |
| `crossdocked` | 10 | 600s | **4** | **1200s** |
| `safety` | 1 | 600s | **1** | **1200s** |

Defaults live in `src/selectbench/cli.py` (`FALLBACK_EXHAUSTIVENESS_CASE`,
`FALLBACK_EXHAUSTIVENESS_CROSSDOCKED`, `FALLBACK_EXHAUSTIVENESS_SAFETY`,
`FALLBACK_TIMEOUT_DEFAULT`). Override per-call by passing
`--fallback_exhaustiveness <N>` / `--fallback_timeout <SEC>` to the
`selectbench` subcommand directly, or disable the retry with `--no-fallback`.

---

## `hardest_safety_targets.py`

Per (method, PoI) ranking of safety panel targets from hardest-to-distinguish
to easiest. With no arguments it processes every
`results/<method>/safety/<poi>.csv` that exists.

Each ranking is emitted at **two scopes** (a `scope` column distinguishes the
rows): `priority` ranks over the 22 priority anti-targets only, `all` ranks
over the full 84-target panel. Ranks restart at 1 within each
(method, PoI, scope).

Outputs:

- `results/<method>/safety/hardest_safety_targets.csv` — per-method
  (columns: `scope, poi, rank, target, sr_1kcal, sr, delta_mean`)
- `results/hardest_safety_targets.csv` — combined across methods (adds `method`)

To rank from an external directory of safety CSVs instead of the default
scan, pass `--from NAME=DIR` (repeatable) — `DIR` may use the nested
`safety/<poi>.csv` layout or a flat `safety_<poi>.csv` layout, labelled as
method `NAME`:

```bash
python scripts/hardest_safety_targets.py \
    --from <method>=/path/to/results_safety_exh1_correct_box/<method>
```

Used by the compose wrappers' `--poi` mode to pick the hardest off-target(s)
automatically when `--safety_target` isn't given; `--hardest_scope` there
selects which scope to consult (default `priority`).
