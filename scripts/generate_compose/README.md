# scripts/generate_compose/

> **You can skip this whole directory if you're evaluating your own model** —
> just drop your molecules into `generated_samples/<method>/<kind>/` and run the
> eval → summarize flow (see [`../README.md`](../README.md)). These wrappers are
> the selectivity-guidance layer we built to benchmark **FKC** and **CFG-SDE** on
> a specific set of pretrained SBDD diffusion models (DiffSBDD, TargetDiff,
> DecompDiff); they patch/drive each baseline's own sampler and are *not* a
> general-purpose tool. They're most useful as a **reference implementation** if
> you want to adapt selectivity guidance onto your own diffusion model — treat
> them as a starting point to port, not a plug-in.

Wrappers for selectivity-guided generation. The model is denoised conditioned
on an **on-target** pocket and pushed away from an **off-target** pocket.
Target selection uses `--case_study`, `--poi`, or `--crossdocked_case` — note
the CrossDocked flag takes an integer case id here, unlike the baseline
wrappers' `--crossdocked <pdb>_<chain>`.

Outputs:

```
generated_samples/<method>_compose/cs/<case>_<tag>_<MODE>.sdf            (--case_study)
generated_samples/<method>_compose/safety/<poi>_<tag>_<MODE>.sdf         (--poi)
generated_samples/<method>_compose/crossdocked/<on>_vs_<off>__<diff>_<tag>_<MODE>.sdf  (--crossdocked_case)
```

`<tag>` is `bon<β_on>_boff<β_off>` when `--beta_on`/`--beta_off` are passed, or
`w<w>` when only `--guidance_weight` is set. `<MODE>` is one of:

- `CFG-SDE` — `--reverse_method SDE` without `--resample`. SDE step with the same CFG combination.
- `FKC` — `--reverse_method SDE --resample`. SDE + Feynman–Kac SMC resampling.
- `DDPM` — `--reverse_method DDPM` (exploratory CFG). Produced and named by the
  wrappers, but **not** picked up by `cross_method.py` (the summary tooling only
  aggregates the `CFG-SDE` and `FKC` modes).

For multi off-target `--poi` runs (`--n_off K` with `K > 1`), the `<poi>` stem gains a `_<K>off_<scope>` marker — e.g. `safety/<poi>_<K>off_<scope>_<tag>_<MODE>.sdf`,
where `<scope>` is the auto-selection panel (`priority` or `all`, from `--hardest_scope`) or `custom` for an explicit `--safety_target` set. The scope is encoded automatically so the summarizer can match each multi-off file to its safety panel — you do **not** need `--run_tag` for this. **`custom` is the exception:** the filename records only that the set was custom, not which targets it held, so `cross_method.py` cannot score those runs against a panel and does not aggregate them (it prints a warning naming each such file). Use `--hardest_scope priority|all` for runs you intend to summarize. The single off-target case (`K = 1`) keeps the plain `<poi>` stem shown above.

---

## Common flags (all three wrappers)

The three wrappers (`gen_compose_diffsbdd.py`, `gen_compose_targetdiff.py`,
`gen_compose_decompdiff.py`) share the same interface for target selection,
guidance, FKC resampling, and off-target selection — documented once here. Each
per-wrapper section below lists **only** what differs (entry point, checkpoint
flag, sampling strategy, and any extra flags).

The invocation pattern is identical across wrappers; only the checkpoint flag
(and, for TargetDiff/DecompDiff, `--config`) changes:

```bash
# CFG-SDE (SDE, no resample)
python scripts/generate_compose/gen_compose_diffsbdd.py --case_study <case> \
    --checkpoint $CKPT --beta_on 2.5 --beta_off -0.5 --reverse_method SDE --device cuda:0

# FKC (SDE + SMC resampling) — add --resample
python scripts/generate_compose/gen_compose_diffsbdd.py --case_study <case> \
    --checkpoint $CKPT --beta_on 2.5 --beta_off -0.5 --reverse_method SDE --resample --device cuda:0

# Negative guidance vs the rank-1 hardest safety target for a PoI
# (resolved from results/<hardest_method>/safety/hardest_safety_targets.csv)
python scripts/generate_compose/gen_compose_diffsbdd.py --poi <poi> \
    --checkpoint $CKPT --beta_on 2.5 --beta_off -0.5 --reverse_method SDE --resample --device cuda:0

# ...or name the off-target explicitly with --safety_target <target>,
# or a CrossDocked pair with --crossdocked_case <id>.
```

**Run every wrapper from the repo root** — each `cd`s into its baseline repo
internally and resolves relative checkpoint/config paths against that dir.

### Target selection & run control

| Argument | Default | Description |
|---|---|---|
| `--case_study` / `--poi` / `--crossdocked_case` | one required | Target selection (`--crossdocked_case` takes an integer case_id) |
| `--device` | `cuda:0` | `cuda:N` or `cpu` |
| `--target_count` | `100` | Valid connected mols to keep in the final SDF |
| `--batch_size` | `50` | Samples per batch |
| `--run_tag` | `""` | Optional stem suffix for exploratory one-offs. The multi-off scope (`priority`/`all`) is encoded automatically, so do **not** use `--run_tag` for that — a `--run_tag` is not read by the standard summary panels. |

### Guidance

| Argument | Default | Description |
|---|---|---|
| `--guidance_weight` (`w`) | `1.0` | Off-target repulsion strength. `0` = on-target only. Shortcut for `(β_on, β_off) = (1+w, -w)`; tags output `w<w>` unless `--beta_*` are set. |
| `--beta_on` / `--beta_off` | none | Signed exponents for `p ∝ p_on^{β_on} · p_off^{β_off}` (general selective form). When both set, override `--guidance_weight` and tag output `bon<x>_boff<y>`. |
| `--reverse_method` | `SDE` | `SDE` (used for CFG-SDE / FKC) or `DDPM` (exploratory CFG; not aggregated). |
| `--inv_temp` | `1.0` | SDE inverse temperature; `>1` sharpens toward the guided distribution. |

### FKC (SMC resampling) — `--resample`

| Argument | Default | Description |
|---|---|---|
| `--resample` | off | Enable SMC resampling (SDE only) — this is what makes a run **FKC** rather than **CFG-SDE** |
| `--resample_thresh` | `0.6` | Active interval — resample only when `t ≥ 1 − resample_thresh` |
| `--ess_thresh` | `0.0` | ESS/N threshold (`0` = resample every step in the interval) |

DiffSBDD and TargetDiff additionally accept `--resample_v` (`1` = positions +
atom types, `0` = positions only, **default `1`**), plus `--dw_clip` and
`--prob_floor` (both default to `0`, i.e. off); TargetDiff also has
`--resample_start`. DecompDiff
supports only the three core FKC flags above.

### Negative-guidance off-target selection (`--poi`)

| Argument | Default | Description |
|---|---|---|
| `--n_off` | `1` | Guide away from the top-N hardest safety targets at once (multi off-target); β_off is split equally across them |
| `--safety_target` | rank-1 hardest | Explicit off-target safety target(s) — one or more names; overrides the auto top-N selection |
| `--hardest_method` | wrapper's own method | Which method's `hardest_safety_targets.csv` to consult for auto-selection |
| `--hardest_scope` | `priority` | Ranking panel for auto-selection: `priority` (22 priority anti-targets) or `all` (full 84-target panel) |

---

## `gen_compose_diffsbdd.py`

Drives DiffSBDD's `compose_ligands.py` inside `baselines/diffsbdd/DiffSBDD` —
apply the compose patch first (see `baselines/diffsbdd/README.md`), since
`compose_ligands.py` ships only in `baselines/diffsbdd/compose_patch/` and does
not exist upstream. **Sampling: batch-retry loop** — keeps sampling batches until
`--target_count` valid connected mols are collected or `--max_batches` is hit.

```bash
python scripts/generate_compose/gen_compose_diffsbdd.py --case_study <case> \
    --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \
    --beta_on 2.5 --beta_off -0.5 --reverse_method SDE --resample --device cuda:0
```

> **Per-batch scratch dir is CWD-relative.** The resume/scratch dir
> (`outputs_compose_<pair>_<tag>/`) is created relative to your current
> directory, so always launch from the repo root — a run started elsewhere
> won't resume an earlier one.

Flags beyond [Common](#common-flags-all-three-wrappers):

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to a DiffSBDD checkpoint (`.ckpt`) |
| `--max_batches` | `10` | Hard cap on retry batches |
| `--timesteps` | model default | Override DDPM/SDE timesteps |
| `--sanitize` | off | Pass-through to DiffSBDD's sanitize flag |
| `--no_rescale_prior` | off | Disable the tempered-prior rescaling. On by default for the SDE path — the initial noise is rescaled by `1/√(inv_temp·(β_on+β_off))`, the prior consistent with the composed score field (no-op for standard CFG, β-sum = 1). |

---

## `gen_compose_targetdiff.py`

Drives the TargetDiff compose patch (`baselines/targetdiff/compose_patch/`) —
apply the patch first (see `baselines/targetdiff/README.md`). Resolves relative
`--checkpoint` / `--config` against the TargetDiff repo. **Sampling:
oversample-once, two-step** — `compose_sample_score.py` samples `--num_samples`
raw molecules to `sample_raw.pt`, then `reconstruct_raw.py` turns it into per-mol
SDFs; re-invocations skip sampling when `sample_raw.pt` already exists.

```bash
python scripts/generate_compose/gen_compose_targetdiff.py --case_study <case> \
    --checkpoint pretrained_models/pretrained_diffusion.pt --config configs/sampling_fkc.yml \
    --beta_on 2.5 --beta_off -0.5 --reverse_method SDE --resample --device cuda:0
```

> The per-run scratch dir (`sample_raw.pt` etc.) is CWD-relative — launch from
> the repo root consistently so re-invocations resume instead of re-sampling.

Flags beyond [Common](#common-flags-all-three-wrappers):

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | (from `--config`) | Overrides `model.checkpoint` in `--config` |
| `--config` | `configs/sampling_fkc.yml` | Sampling config yml |
| `--num_samples` | `150` | Raw molecules to sample (oversample; TargetDiff fragments ~20–30%) |
| `--num_steps` | `1000` | Diffusion steps |
| `--seed` | `2021` | RNG seed |
| `--sample_num_atoms` | `prior` | `prior` (pocket-conditioned) or `fixed` |
| `--num_atoms_to_sample` | `-1` | Fixed ligand atom count (required when `--sample_num_atoms fixed`) |
| `--uniform_num_atoms` | off | One atom count per batch — **auto-enabled** with `--resample` in `prior` mode (SMC needs equal sizes) |
| `--resample_start` | `0.0` | Skip resampling during the earliest fraction of denoising |
| `--no_rescale_prior` | off | Disable the tempered-prior rescaling (same semantics as DiffSBDD; no-op for standard CFG) |

---

## `gen_compose_decompdiff.py`

Drives the DecompDiff compose patch (`baselines/decompdiff/compose_patch/`) —
apply the patch first (see `baselines/decompdiff/README.md`). Resolves relative
`--ckpt_path` / `--config` against the DecompDiff repo. **Sampling:
oversample-once, two-step** — `prepare_custom_data.py` builds a one-entry LMDB
(pocket PDB + reference ligand SDF → Murcko `ref_prior` decomposition), then
`compose_for_pockets.py` samples and writes per-mol SDFs directly (reconstruction
is built in); re-invocations skip the LMDB build / sampling when their outputs
already exist.

> **Per-batch scratch dir is CWD-relative.** As with the DiffSBDD and
> TargetDiff wrappers, the resume/scratch dir
> (`outputs_compose_<pair>_<tag>/`) is created relative to your current
> directory, so always launch from the repo root — a run started elsewhere
> won't resume an earlier one.

```bash
python scripts/generate_compose/gen_compose_decompdiff.py --case_study <case> \
    --ckpt_path pretrained_models/uni_o2_bond.pt --config configs/sampling_drift.yml \
    --beta_on 2.5 --beta_off -0.5 --reverse_method SDE --resample --device cuda:0
```

Flags beyond [Common](#common-flags-all-three-wrappers):

| Argument | Default | Description |
|---|---|---|
| `--ckpt_path` (alias `--checkpoint`) | (from `--config`) | Checkpoint path; overrides `model.checkpoint` in `--config` |
| `--config` | `configs/sampling_drift.yml` | Sampling config yml |
| `--num_samples` | `150` | Raw molecules to sample (oversample for fragmentation) |
| `--recon_with_bond` | `"True"` | Reconstruct with bond info during decoding (string passed straight through to `compose_for_pockets.py`) |
| `--prior_mode` | `ref_prior` | Decomposed prior — `ref_prior` (reference-ligand-derived, matches the unguided baseline) or `subpocket` (protein-pocket-derived). **Also drives the tempered-prior rescaling**: `ref_prior` runs with rescaling on, `subpocket` with it off — they are coupled (a `prior_mode × rescale` sweep showed the mismatched combinations produce only fragmented molecules), so there is no separate `--no_rescale_prior` knob for DecompDiff. |

DecompDiff does **not** take the atom-count flags or the extra FKC knobs
(`--resample_v`, `--dw_clip`, `--prob_floor`, `--resample_start`).

---

## Choosing a CrossDocked pair

The compose wrapper takes one oriented (on, off) pair at a time via
`--crossdocked_case <case_id>`. There's no batch launcher — pick a case_id and
run the wrapper, which keeps each output SDF tied to exactly one CSV row. Run
these from the **repo root** (as elsewhere in this README).

**Specific pair by id.** Each row of
`data/crossdocked_test_set/crossdocked_test_set_pairs.csv` is one oriented
pair; its `case_id` is the value to pass. `on_target` names which side is the
on-target (the other is the off-target), and the wrapper resolves both pockets
from that row:

```bash
python scripts/generate_compose/gen_compose_diffsbdd.py --crossdocked_case 0 \
    --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \
    --beta_on 2.5 --beta_off -0.5 --reverse_method SDE --resample --device cuda:0
```

**Random pair(s) at a difficulty.** `_gen_common.list_crossdocked_pairs`
returns `(case_id, on_name, off_name)` tuples for a tier and can
deterministically subsample them:

```bash
# 5 random "easy" pairs, reproducibly → prints "case_id on off"
python -c "import sys; sys.path.insert(0, 'scripts/generate_compose'); \
from _gen_common import list_crossdocked_pairs; \
[print(c, on, off) for c, on, off in list_crossdocked_pairs('easy', 5, seed=42)]"
```

Difficulty is one of `easy`, `medium`, `hard`, or `all`. Drop the `n_samples`
argument to list the whole tier. Feed each printed `case_id` to the wrapper as
`--crossdocked_case`. To sweep a tier, loop in your shell:

```bash
for cid in $(python -c "import sys; sys.path.insert(0,'scripts/generate_compose'); \
from _gen_common import list_crossdocked_pairs; \
print(' '.join(str(c) for c, _, _ in list_crossdocked_pairs('hard')))"); do
    python scripts/generate_compose/gen_compose_diffsbdd.py --crossdocked_case "$cid" \
        --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \
        --beta_on 2.5 --beta_off -0.5 --reverse_method SDE --resample --device cuda:0
done
```

---

## Negative-guidance pipeline (PoI ⟂ hardest safety target)

For FKC/CFG generation that steers **away** from the safety-panel target the
baseline is worst at distinguishing:

```bash
# Stage 1-2: baseline generation + safety eval on each PoI
#   (unguided DiffSBDD baseline via upstream generate_ligands.py — see baselines/diffsbdd/README.md §2)
python scripts/run_eval.py --method <method> --eval safety

# Stage 3: rank safety targets by difficulty (both priority + all scopes)
python scripts/hardest_safety_targets.py
# → writes results/<method>/safety/hardest_safety_targets.csv

# Stage 4: guided generation against rank-1 hardest off-target
#   --hardest_scope priority (default) ranks over the 22 priority anti-targets;
#   pass --hardest_scope all to rank over the full 84-target panel.
python scripts/generate_compose/gen_compose_diffsbdd.py --poi <poi> \
    --checkpoint baselines/diffsbdd/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt \
    --beta_on 2.5 --beta_off -0.5 --reverse_method SDE --resample --device cuda:0
```

`hardest_safety_targets.py` lives in `scripts/` (not here) — see
`scripts/README.md`.
