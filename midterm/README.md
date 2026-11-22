# Midterm: autoencoder and SimCLR representations on Tiny-ImageNet

Representation Learning 2026. Part A (denoising autoencoder) is implemented and evaluated.
Part B (contrastive losses and augmentations) is in progress; its section at the end lists what is missing.

The code follows the course `lec4/` layout: a flat folder run with `python train.py`, LightningModules in
`model.py`, a LightningDataModule in `datamodule.py`, callbacks, losses and augmentations in their own files.
The one structural difference: every run reads a YAML config (`--config`), as the assignment requires, instead
of the module constants in `lec4/config.py`.

## Setup

Python 3.12. The environment is managed with [uv](https://docs.astral.sh/uv/) from the repo root:

```bash
git clone -b midterm <your fork url> && cd <repo>
uv sync                  # installs everything in pyproject.toml
cd midterm
```

Data downloads itself on first use and is cached in `midterm/dataset/` (not committed): Tiny-ImageNet from Hugging Face
(`zh-plus/tiny-imagenet`, about 160 MB) and CIFAR-100 from torchvision. A CUDA GPU is optional but real runs need one;
the free Colab T4 is enough.

Check that everything is wired up (CPU, about a minute):

```bash
uv run python train.py --config configs/smoke.yaml            # 2-epoch DAE on 2% of the data
uv run python train.py --config configs/eval_random_smoke.yaml
```

Each run prints `Final metrics: {...}`, writes `results/<run_name>.json` and logs to `outputs/tb_logs/<run_name>/`.

## Running one experiment

```bash
uv run python train.py --config configs/partA/final/final_dae.yaml
```

A config lists only what differs from `config.py` (`DEFAULTS`); an unknown key raises an error. The `mode` key picks what runs:

| mode | what it does |
|---|---|
| `pretrain_dae` | trains the denoising autoencoder, saves the encoder to `outputs/encoders/<run_name>.pt` |
| `evaluate` | frozen encoder: weighted cosine k-NN and linear probe (3 probe seeds) |
| `finetune` | trains all weights on 1%, 10% or 100% of the labels (also the from-scratch baseline when `encoder_path` is empty) |
| `robustness` | frozen encoder: probe and k-NN accuracy on corrupted evaluation images |

Training resumes from `outputs/checkpoints/<run_name>/last.ckpt` if it exists (Colab can disconnect).
All seeds are fixed (`seed: 42`; fine-tuning repeats use 43 and 44).

## Reproducing Part A, in order

The configs of a step sort so that training comes before evaluation. Later steps need the encoders saved by
`configs/partA/final/final_dae.yaml` and `final_ae.yaml`, so run the steps in this order. This loop skips runs that
already have a result file:

```bash
for f in configs/partA/search/*.yaml; do
  [ -f results/$(basename $f .yaml).json ] || uv run python train.py --config $f
done
```

| step | folder | runs | purpose |
|---|---|---|---|
| 1 | `configs/partA/search/` | 30 | random search over corruption type and strength, latent size, learning rate, weight decay (proxy: 25% of train, 20 epochs), picked by validation k-NN |
| 2 | `configs/partA/seedcheck/` | 6 | the top trials again with 2 more seeds |
| 3 | `configs/partA/final/` | 8 | full DAE and vanilla AE, trained until validation reconstruction error stops improving (cap 100 epochs), then k-NN and probe on both layers and a random-init encoder |
| 4 | `configs/partA/finetune/` | 18 | fine-tuning at 1/10/100% labels, both layers, scratch vs AE vs DAE. No fixed epoch count: each run stops when validation top-1 has not improved for a few epochs and keeps its best epoch |
| 5 | `configs/partA/finetune_extra/` | 34 | 2 extra seeds on the pooled conv layer, plus a learning-rate and weight-decay grid at 10% labels |
| 6 | `configs/partA/transfer/` | 33 | the same encoders on CIFAR-100 (resized to 64x64): frozen evaluation and 3-seed fine-tuning |
| 7 | `configs/partA/robustness/` | 3 | probe and k-NN under noise, blur, low contrast, JPEG and patch masking, severities 1-5 |
| 8 | `configs/partA/test/` | 66 | the final numbers on the test split: frozen evaluation (both layers) and 3-seed fine-tuning (pooled conv), Tiny-ImageNet and CIFAR-100. Run once, after everything above is fixed |

`make_configs.py` writes all of these folders (`uv run python make_configs.py`), so any grid can be regenerated.
The chosen DAE (patch masking, rate 0.30, latent size 512, learning rate 1e-3, weight decay 0) is set in
`make_full_runs`.

Data splits: 90k train images, a 10k hold-out (50 per class) for all tuning, and the official validation split
(10k) as the test set. Runs use `eval_split: val`; the test split is only used for final numbers.

### Analysis scripts

Run from `midterm/`:

```bash
uv run python -m scripts.summarize      # results/*.json -> summaries/*.csv (every table and plot reads these)
uv run python -m scripts.interpret      # per-class k-NN, t-SNE, retrieval grid, shuffled-label probe check
uv run python -m scripts.scale_check    # why the DAE probe breaks under damage (feature shift)
```

`scripts/analysis.py` holds the plot functions (reconstruction error vs k-NN, label efficiency, robustness).
The figures are in `figures/`. `summaries/` holds the aggregated numbers behind them as small CSV files.
The encoders and the raw `results/*.json` files are not committed; `summarize` rebuilds the CSVs from them.

## Running on Colab

`notebooks/colab_launcher.ipynb` only mounts Drive, clones the repo, installs `requirements.txt` and calls
`train.py`. Use a T4 runtime. Outputs and results are kept on Drive, so after a disconnect, run the cells again
and training resumes. To run a folder of configs, set `CONFIG_GLOB` in the loop cell (finished runs are skipped).
TensorBoard: `tensorboard --logdir outputs/tb_logs`.

## Files

| file | role |
|---|---|
| `train.py` | entry point: loads the config, dispatches the mode |
| `config.py` | default values and the YAML loader |
| `utils.py` | `parse_args`, `build_model` |
| `model.py` | encoder, decoder, `DenoisingAutoEncoder`, `EncoderClassifier` |
| `datamodule.py` | Tiny-ImageNet and CIFAR-100 DataModules, splits, balanced label subsets |
| `augmentations.py` | lec4 transforms plus the input corruptions used by the DAE and the robustness test |
| `losses.py` | lec4 file, not yet changed (Part B) |
| `callbacks.py` | progress bar and the k-NN monitor used for model selection |
| `evaluation.py` | feature extraction, k-NN, linear probe, `run_evaluation` |
| `robustness.py` | the robustness test (`mode: robustness`) |
| `make_configs.py` | writes the config grids |
| `scripts/` | analysis and plotting helpers |

## Part B

Not finished: the NT-Xent, NT-Logistic and margin-triplet losses, GPU augmentations, the SimCLR module, the
collapse metrics and the augmentation ablation are still to be added to `losses.py`, `augmentations.py` and
`model.py`. This section will list its configs and run order once they exist.
