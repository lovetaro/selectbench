# Baselines

Per-baseline setup and generation instructions. Each subdirectory has its own
README covering env setup, checkpoint download, and how to generate samples.

| Baseline | Upstream |
|---|---|
| [DiffSBDD](diffsbdd/) | [arneschneuing/DiffSBDD](https://github.com/arneschneuing/DiffSBDD) |
| [TargetDiff](targetdiff/) | [guanjq/targetdiff](https://github.com/guanjq/targetdiff) |
| [DecompDiff](decompdiff/) | [bytedance/DecompDiff](https://github.com/bytedance/DecompDiff) |
| [Pocket2Mol](pocket2mol/) | [pengxingang/Pocket2Mol](https://github.com/pengxingang/Pocket2Mol) |
| [PocketFlow](pocketflow/) | [Saoge123/PocketFlow](https://github.com/Saoge123/PocketFlow) |
| [MolFORM](molform/) | [huang3170/MolForm](https://github.com/huang3170/MolForm) |
| [MolSnapper](molsnapper/) | [oxpig/MolSnapper](https://github.com/oxpig/MolSnapper) |
| [CMD-GEN](cmdgen/) | [zyrlia1018/CMD-GEN](https://github.com/zyrlia1018/CMD-GEN) |
| [SpecLig](speclig/) | [CQ-zhang-2016/SpecLig](https://github.com/CQ-zhang-2016/SpecLig) |

## Reproducibility & pinned versions

Because we clone from upstream repositories, package and config drift is a real risk. To guard against it:

- **Pinned upstream commits.** Where we ship a compose patch (DiffSBDD,
  TargetDiff, DecompDiff), each `compose_patch/README.md` names the exact
  upstream commit the patch was authored against, and the setup steps
  `git checkout` that commit rather than tracking `main`. The patches add
  compose methods to the upstream classes at that commit. They are strictly
  additive for DiffSBDD's two compose files; TargetDiff and DecompDiff also
  modify a small number of upstream lines (TargetDiff: `sample_diffusion` and
  `sample_diffusion_ligand` signatures; DecompDiff: the `torch_sparse` removal
  in `uni_transformer_edge.py`). Each `compose_patch/README.md` states what its
  patch changes.
- **Environments.** Each baseline is run in its upstream-provided conda
  environment, except DecompDiff, which ships none upstream and uses the manual
  install in its README (env name and key pinned versions are noted in each
  subdirectory's README). We activate the env name that the upstream
  `environment.yaml`/`env.yml` actually declares.
- **Default inference settings.** For every baseline we follow the upstream
  **default inference/sampling configuration** — we do **not** retrain any
  model and do not tune sampling hyperparameters for the unguided baselines.
  Each README records the checkpoint used and any flags passed (e.g. DiffSBDD
  `--sanitize`). Only the compose (selectivity-guidance) variants add flags,
  all documented in `scripts/generate_compose/README.md`; they reuse the same
  pretrained checkpoints unchanged.

Pinned upstream commits, per-baseline conda environments, and the exact
checkpoints and flags used are documented in each subdirectory's README (and,
for the compose variants, in `scripts/generate_compose/README.md`). We do not
re-host upstream model checkpoints — each README links the upstream download —
with one exception: TargetDiff's checkpoint is vendored under
`baselines/targetdiff/pretrained_models/` because all three of upstream's
download links now 404 (provenance, licence and sha256 are documented there).


**Three diffusion baselines ship an optional first-party selective-generation framework** — `scripts/generate_compose/gen_compose_{diffsbdd,targetdiff,decompdiff}.py` — which steers sampling toward the on-target and away from the off-target(s). It is a reproduction convenience, *not* required to benchmark your own model (see [`data/README.md` → "Using these targets with your own model"](../data/README.md#using-these-targets-with-your-own-model)).
Each compose wrapper reuses the pretrained checkpoint unchanged and applies an *additive* patch to the upstream sampler (see the per-baseline `compose_patch/README.md`). Coverage differs by baseline:

- **DecompDiff** — ships both an unguided wrapper (`baselines/decompdiff/gen_decompdiff.py`) and the compose wrapper. It's the one baseline that *needs* an unguided wrapper: upstream has no arbitrary-pocket entry point (see `baselines/decompdiff/README.md`).
- **DiffSBDD** — ships the compose wrapper; its unguided baseline runs upstream `generate_ligands.py` directly (see `baselines/diffsbdd/README.md`).
- **TargetDiff** — ships the compose wrapper; its unguided path follows the upstream repo (see `baselines/targetdiff/`).

`generated_samples/` and `results/` are gitignored working dirs (the generate → evaluate → summarize flow reproduces them on demand); `results/README.md` documents their layout and column schema, and shows how to build a small filled-in `demo` example to anchor your own output against.

For every other baseline we vendor no model checkpoints and no generated
samples. The only vendored *code* outside the three `compose_patch/` trees is
MolFORM's patched sampler (`baselines/molform/patch/`) and CMD-GEN's GMM
wrapper (`baselines/cmdgen/cmdgen_gmm.py`); both are listed as third-party
components in the root `LICENSE`.
Instead, each subdirectory's README points to the upstream repository, documents environment setup, and specifies the (single) filesystem convention SelectBench expects: drop the resulting `.sdf` (or `.smi`) under `generated_samples/<method>/{cs,safety,crossdocked}/`, and the evaluation and summarisation pipeline auto-discovers it without any further configuration.

## Benchmarking your own model

Nothing under `baselines/` is required to benchmark your own model — the input
pockets to condition on, target enumeration, and output-file naming live in
[`data/README.md` → "Using these targets with your own model"](../data/README.md#using-these-targets-with-your-own-model),
and the evaluate → summarize workflow is in
[`scripts/README.md`](../scripts/README.md). In short: drop SDFs into
`generated_samples/<method>/{cs,safety,crossdocked}/`, then

```bash
python scripts/run_eval.py --eval cs        # or safety / crossdocked
python scripts/summarize/cross_method.py
```
