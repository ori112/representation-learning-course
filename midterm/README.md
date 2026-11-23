# Midterm: autoencoder and SimCLR representations on Tiny-ImageNet

Representation Learning 2026. Part A (denoising autoencoder) is implemented and evaluated.
Part B (contrastive losses and augmentations) is in progress. Its section at the end says what is missing.

The code follows the course `lec4/` layout: a flat folder run with `python train.py`, LightningModules in
`model.py`, a LightningDataModule in `datamodule.py`, callbacks, losses and augmentations in their own files.
One structural difference: every run reads a YAML config (`--config`), as the assignment requires, instead of
the module constants in `lec4/config.py`.

## Three ways to check this work, from cheapest to fullest

| | what you do | needs | time |
|---|---|---|---|
| 1. Wiring test | run two tiny configs | a laptop, no GPU | about 3 minutes |
| 2. Rebuild every table and figure | `summarize` + `make_figures` on the committed raw results | a laptop, no GPU | about 1 minute |
| 3. Rerun everything from zero | open the Colab notebook and run its cells | a Google account, free T4 GPU | training alone adds up to about 4.5 hours, plus evaluation and Colab overhead. Restartable (see below) |

### Setup (needed for 1 and 2)

Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
git clone -b midterm https://github.com/ori112/representation-learning-course.git
cd representation-learning-course
uv sync              # installs everything in pyproject.toml
cd midterm
```

### 1. Wiring test

```bash
uv run python train.py --config configs/smoke.yaml            # 2-epoch DAE on 2% of the data
uv run python train.py --config configs/eval_random_smoke.yaml
```

Tiny-ImageNet downloads itself on first use (Hugging Face `zh-plus/tiny-imagenet`, about 160 MB) and is cached
in `midterm/dataset/` (not committed). Each run prints `Final metrics: {...}`, writes `results/<run_name>.json`
and logs to `outputs/tb_logs/<run_name>/`.

### 2. Rebuild the tables and figures from the committed results

The raw result of every run of the final pipeline is committed in `final_results/` (one small JSON per run).

```bash
uv run python -m scripts.summarize --results final_results   # JSON -> summaries/*.csv
uv run python -m scripts.make_figures                        # summaries/*.csv -> figures/*.png
```

### 3. Rerun the whole pipeline on Colab

Open the launcher notebook (it loads straight from GitHub, never from a saved copy):

<https://colab.research.google.com/github/ori112/representation-learning-course/blob/midterm/midterm/notebooks/colab_launcher.ipynb>

Use a GPU runtime (Runtime > Change runtime type > T4 GPU). Run the first two code cells (they mount your own
Google Drive, clone the repo, install `requirements.txt`, and keep outputs in a `rl_midterm/` folder on your Drive),
then the cell `!python run_plan.py`. It runs the whole of Part A, one config per run through `train.py`, in order.
If Colab disconnects, run the first two cells and the plan cell again: finished runs are skipped and an
interrupted run resumes from its checkpoint. The last cell, `!python run_plan.py --only test`, runs the final
test-set numbers once, after the rest has finished.

To run a single experiment from a terminal instead:

```bash
uv run python train.py --config configs/partA/search/dae_search_00.yaml
```

## What the plan does, in order (`run_plan.py`)

Nothing about the final model is typed in by hand. `make_configs.py` writes the next configs from the results of
the previous step.

| step | configs | runs | what it does |
|---|---|---|---|
| search | `configs/partA/search/` | 30 | random search over corruption type and strength, latent size, learning rate, weight decay (proxy: 25% of train, 20 epochs). k-NN on the validation hold-out is measured on both candidate layers, and the pooled conv layer is h |
| seedcheck | `configs/partA/seedcheck/` | 6 | the 3 best trials by validation k-NN, rerun with 2 more seeds (written by `make_configs.py --stage seedcheck`) |
| pretrain | `configs/partA/final/final_*.yaml` | 6 (DAE and vanilla AE, 3 seeds) | the final DAE (best 3-seed mean, written to `configs/partA/final/selection.json`) and its vanilla twin (strength 0). Both train until validation reconstruction error stops improving, and keep their best epoch |
| frozen | `configs/partA/final/full_eval_*.yaml` | 18 | k-NN and linear probe (3 probe seeds) on both layers, for DAE, vanilla AE and a random-init encoder |
| robustness | `configs/partA/robustness/` | 9 (3 models x 3 seeds) | probe and k-NN under noise, blur, low contrast, JPEG and patch masking, severities 1-5 |
| tune | `configs/partA/tune/` | 18 | learning-rate and weight-decay grid at 10% labels for scratch, vanilla AE and DAE. The best setting per start is picked from validation (`configs/partA/tune/selection.json`) |
| finetune, seeds | `configs/partA/finetune/`, `finetune_extra/` | 9 + 18 | fine-tuning at 1/10/100% labels (3 seeds), scratch vs vanilla AE vs DAE. No fixed epoch count: a run stops when validation top-1 has not improved for a few epochs and keeps its best epoch |
| transfer | `configs/partA/transfer/` | 45 | the same encoders on CIFAR-100 (resized to 64x64): frozen evaluation and 3-seed fine-tuning |
| test | `configs/partA/test/` | 90 | the final numbers on the test split, same recipes, run once with `--only test` |

Data splits: 90k train images, a 10k hold-out (50 per class) for every tuning and stopping decision, and the official
validation split (10k) as the test set. Everything before the last step uses the hold-out. All seeds are fixed
(`seed: 42`, and repeats use 43 and 44).

### Analysis scripts (run from `midterm/`, after `summarize`)

```bash
uv run python -m scripts.make_figures   # recon-vs-kNN, layer choice, label efficiency, robustness
uv run python -m scripts.interpret      # per-class k-NN, t-SNE, retrieval grid, shuffled-label probe check
uv run python -m scripts.scale_check    # why the DAE probe breaks under damage (feature shift)
```

`interpret` and `scale_check` need the trained encoders (`outputs/encoders/final_*.pt`, produced by the pretrain step,
so copy them from the Colab Drive folder). TensorBoard logs of every run are written to `outputs/tb_logs/` (on Colab, the
`rl_midterm` Drive folder). View them with `tensorboard --logdir outputs/tb_logs`.

The TensorBoard logs of the final Part A run (one folder per run) are shared here:
<https://drive.google.com/drive/folders/1rdpXOjrG7_i45iYjwYg-IHhgvJTU6gVE>. Download the folder and point
`tensorboard --logdir` at it.

## Files

| file | role |
|---|---|
| `train.py` | entry point: loads the config, dispatches the mode |
| `run_plan.py` | runs the whole Part A plan in order |
| `config.py` | default values and the YAML loader |
| `utils.py` | `parse_args`, `build_model` |
| `model.py` | encoder, feature layers, decoder, `DenoisingAutoEncoder`, `EncoderClassifier` |
| `datamodule.py` | Tiny-ImageNet and CIFAR-100 DataModules, splits, balanced label subsets |
| `augmentations.py` | lec4 transforms plus the input corruptions used by the DAE and the robustness test |
| `losses.py` | lec4 file, not yet changed (Part B) |
| `callbacks.py` | progress bar and the k-NN monitor used for model selection |
| `evaluation.py` | feature extraction, k-NN, linear probe, `run_evaluation` |
| `robustness.py` | the robustness test (`mode: robustness`) |
| `make_configs.py` | writes the config grids, stage by stage |
| `scripts/` | `summarize`, `make_figures`, `analysis`, `interpret`, `scale_check`, `export_curves` |

`mode` in a config picks the job: `pretrain_dae`, `evaluate`, `finetune` or `robustness`. A config lists only what
differs from `config.py` (`DEFAULTS`), and an unknown key raises an error.

## Part B

Not finished: the NT-Xent, NT-Logistic and margin-triplet losses, GPU augmentations, the SimCLR module, the
collapse metrics and the augmentation ablation are still to be added to `losses.py`, `augmentations.py` and
`model.py`. This section will list its configs and run order once they exist.
