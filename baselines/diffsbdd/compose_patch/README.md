# DiffSBDD compose patch

Drop-in files that add selectivity-guided ("compose") sampling to upstream
[DiffSBDD](https://github.com/arneschneuing/DiffSBDD) (Schneuing et al., 2024).
Required by `scripts/generate_compose/gen_compose_diffsbdd.py`; not needed
for the unguided baseline (upstream `generate_ligands.py`; see
`baselines/diffsbdd/README.md`).

## Apply

After cloning upstream DiffSBDD per `baselines/diffsbdd/README.md` step 1,
copy these files into the cloned repo:

```bash
# from selectbench root, after baselines/diffsbdd/DiffSBDD exists
cp baselines/diffsbdd/compose_patch/compose_ligands.py \
   baselines/diffsbdd/compose_patch/lightning_modules.py \
   baselines/diffsbdd/DiffSBDD/
cp baselines/diffsbdd/compose_patch/equivariant_diffusion/*.py \
   baselines/diffsbdd/DiffSBDD/equivariant_diffusion/
```

Then `scripts/generate_compose/gen_compose_diffsbdd.py` works as documented
in `scripts/generate_compose/README.md`.

## What changes

These files are drop-in additions over upstream `DiffSBDD` at commit
**`5d0d38d`** (`5d0d38d16c8932a0339fd2ce3f67ade98bbdff27`) — the commit the
setup step in [`../README.md`](../README.md) pins. Verified against that commit:
the additions are pure extensions of the upstream classes (0 upstream lines
removed), so no upstream method is modified. Running the unguided baseline path
(`generate_ligands.py`) is identical in behavior whether or not this patch is
applied.

| file | purpose |
|---|---|
| `compose_ligands.py` | compose sampling entry point |
| `equivariant_diffusion/conditional_model.py` | adds `compose_sample_given_pocket` and `_compose_step` to `ConditionalDDPM` |
| `equivariant_diffusion/en_diffusion.py` | adds `_gamma_derivative` helper to `EnVariationalDiffusion` (computes dγ/dt for the SDE step used by both CFG-SDE and FKC) |
| `lightning_modules.py` | biopython compatibility shim — *not compose-specific* |

## Scope notes

- `_compose_step` implements both reverse-process variants used in the paper:
  CFG-SDE (no resampling) and FKC (Feynman-Kac systematic resampling).
  Selection is driven by the `--reverse_method` and `--resample` flags on
  `compose_ligands.py`.

## License

Upstream DiffSBDD is MIT licensed; the original copyright notice is
reproduced in `LICENSE.upstream` (covering all files copied from upstream
in this directory).
