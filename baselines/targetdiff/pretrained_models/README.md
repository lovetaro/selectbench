# Vendored TargetDiff checkpoint

`pretrained_diffusion.pt` is TargetDiff's CrossDocked-trained diffusion
checkpoint, redistributed here **only because upstream's own download links are
dead** (see below). It is not a SelectBench artifact and we did not train it.

| | |
|---|---|
| Upstream | [guanjq/targetdiff](https://github.com/guanjq/targetdiff) @ `142f1eb` |
| Author | Jiaqi Guan et al. |
| License | MIT — see `../LICENSE.upstream` |
| Size | 32 MB |
| sha256 | `8977821339ba31b7103b46835ef4ddbf7f15f3f2efc5cb1f2a405e7905b09db6` |

Verify after checkout:

```bash
sha256sum baselines/targetdiff/pretrained_models/pretrained_diffusion.pt
```

## Why it is vendored

Upstream's README links its weights from three Google Drive folders, and as of
2026-09 **all three return HTTP 404**:

```
drive.google.com/drive/folders/1-ftaIrTXjWFhw3-0Twkrs5m0yX6CNarz  -> 404
drive.google.com/drive/folders/19imu-mlwrjnQhgbXpwsLgA17s1Rv70YS  -> 404
drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK  -> 404
```

This is not a network artifact: five other SelectBench baselines host their
checkpoints on Google Drive and all resolve normally. Upstream's `main` is
still at the pinned commit, so the links have not been restored.

Prefer upstream if it is ever fixed — this copy exists so the TargetDiff
baseline stays reproducible in the meantime.

## Use

The sampler resolves `--checkpoint` relative to the clone, so copy or symlink
it into place after cloning:

```bash
cd baselines/targetdiff/targetdiff
mkdir -p pretrained_models
ln -s ../../pretrained_models/pretrained_diffusion.pt \
      pretrained_models/pretrained_diffusion.pt
```
