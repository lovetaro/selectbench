# MolFORM patch

At the pinned upstream commit (`huang3170/MolForm@7a05ede`) the single-pocket
sampling entry point `scripts/sample_for_pocket.py` is **out of sync** with the
rest of the repo and cannot run as shipped. This directory holds a corrected
drop-in replacement.

## Why it's needed

`scripts/sample_for_pocket.py` at `7a05ede` does not match the tested code path
in `scripts/sample_diffusion.py`. Four defects, each independently fatal:

1. **DataParallel prefix not stripped.** The released checkpoint
   (`molform_dpo_model.pt`) is saved from a `DataParallel` model, so every key is
   `module.<name>`. The upstream script does a strict `load_state_dict` and fails
   with a missing/unexpected-keys error. (`scripts/sample_diffusion.py` strips the
   prefix; this script does not.)
2. **Wrong `sample_diffusion_ligand` signature.** The upstream call passes
   `num_steps=` (the real parameter is `num_step`) and `pos_only=` (not a
   parameter at all), so the call raises `TypeError`.
3. **Wrong return arity.** It unpacks 7 return values; the function returns 4.
4. **Missing `argmax`.** `sample_diffusion_ligand` returns per-atom type
   *distributions*; the script feeds them straight into reconstruction without
   `np.argmax(axis=1)`, raising `TypeError: unhashable type: 'list'`.

## What the fix is

`sample_for_pocket.py` in this directory is the upstream file with **only** those
four defects corrected — every fix mirrors what `scripts/sample_diffusion.py`
already does. All model / sampling / reconstruction logic is upstream's own,
copied verbatim. No retraining, no change to model behaviour.

With the fix the script runs against the **released** `molform_dpo_model.pt` and
the **unmodified** upstream `configs/sampling.yml` — no `module.`-stripped copy of
the checkpoint and no config edits are required.

## How to apply

After cloning + checking out `7a05ede` (see [`../README.md`](../README.md)),
copy this file over the upstream script in the clone:

```bash
cp baselines/molform/patch/sample_for_pocket.py \
   baselines/molform/MolForm/scripts/sample_for_pocket.py
```

Then follow the generation step in [`../README.md`](../README.md).
