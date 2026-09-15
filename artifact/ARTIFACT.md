# LFreeDA Artifact (Scaled-Down)

## Table of Contents

- [Scope](#scope)
- [What this artifact reproduces](#what-this-artifact-reproduces)
- [Environment](#environment)
- [Data required](#data-required)
- [How to run](#how-to-run)
  - [Step I: Pseudo-label generation](#step-i-pseudo-label-generation-9-min)
  - [Step II: Pseudo-label selection](#step-ii-pseudo-label-selection-10-min)
  - [Step III: Adaptation with selected pseudo-labels](#step-iii-adaptation-with-selected-pseudo-labels-79-hrs)
- [Expected results](#expected-results)

## Scope

This artifact provides a **scaled-down, fully reproducible demonstration of LFreeDA on a single adaptation task: July → August 2024**. The paper's full evaluation spans five rolling adaptation tasks (July→Aug, Aug→Sep, Sep→Oct, Oct→Nov, Nov→Dec), each taking ~9 hours end-to-end on a single GPU (see per-step breakdown below) — ~45 GPU-hours total, which isn't practical within a typical artifact evaluation window.

Running this artifact fully reproduces one representative task and lets you independently verify the corresponding numbers reported in the paper. Code for all five tasks is in the main repository, documented in the top-level [readme.md](../readme.md) — provided for future research, but **not** part of this artifact's evaluation scope.

## What this artifact reproduces

- **Figure 3** (Step I performance) — for the July→Aug task only and for LFreeDA's Step I method only.
- **Table 6** (Step II accuracy, coverage, and ACS for five outlier detection methods) — for the July→Aug task only (under the "Aug testing" column). We additionally report pseudo-label accuracy under two settings described in the paper (original / confidence filtering only). Note that Figure 5 in the paper reports this **averaged across all five tasks**, so it isn't directly comparable to a single-task rerun — see the July→Aug-specific reference values under [Expected results](#expected-results) below instead.
- **Figure 6** (Step III adaptation results, all DA methods) — for the July→Aug task only.

## Environment

See [Hardware Specifications](../readme.md#hardware-specifications) and [Installation](../readme.md#installation) in the main README — this artifact uses the same environment and setup as the full reproduction (only the scope differs).

## Data required

This artifact ships the data needed for the July → August task:
- `data/graph_features/mb24/{July,Aug}`
- `data/image_features/mb24/{july,aug}`
- `data/labels/mb24/{July,Aug}`
- `data/stepI_trained_models/mb24/aug/` — trained Step I model, used by default
- `data/stepII_constructed_datasets/mb24/aug/` — precomputed Step II output, used by default

We include these precomputed intermediate outputs so each step can be validated independently by default, which makes review simpler and less fragile to any single step's variance.

## How to run

### Step I: Pseudo-label generation (~9 min)
```bash
python StepI/train_mb24+_aug.py --epochs 30
```
Trains the Step I model and reports test accuracy for the July→Aug task. 

<details>
<summary><strong>Optional: save your own trained model</strong></summary>

The save block at the end of the script is commented out by default. Uncomment it if you'd also like to save the trained model to `results/stepI_trained_models_scratch/mb24/aug/`.

</details>

<details>
<summary><strong>Optional: evaluate the precomputed model directly (no training, deterministic)</strong></summary>

```bash
python StepI/evaluate_pretrained_aug.py
```
Loads `data/stepI_trained_models/mb24/aug/` (the same model Step II uses by default) and reports its `Acc test target`/`Macro F1 test target` on the July→Aug task's test set, without training anything — useful as a deterministic sanity check against the expected values below, independent of training-run variance.

</details>

### Step II: Pseudo-label selection (~10 min)
```bash
jupyter notebook StepII/mb24+/mb24+_aug.ipynb
```
By default, the notebook loads the trained Step I model directly from `data/stepI_trained_models/mb24/aug/`. You can just run all cells. This reproduces Table 6 (accuracy, coverage, and ACS for five outlier detection methods, July→Aug task) and computes pseudo-label accuracy under two additional settings described in the paper (original / confidence filtering only) — the setting Figure 5 in the paper reports as an average across all five tasks, so compare your July→Aug numbers against the reference values under [Expected results](#expected-results) rather than Figure 5 directly. We've kept the cell outputs saved in the notebook so you can compare against the expected results.

<details>
<summary><strong>Optional: evaluate a different Step I model, or save the constructed datasets</strong></summary>

- To evaluate a different Step I model, point the model-loading cell at `results/stepI_trained_models_scratch/mb24/aug/{generator,classifier}`.
- To also save the constructed datasets, uncomment the save block in the last cell — output goes to `results/stepII_constructed_datasets_scratch/mb24/aug/`.

</details>

### Step III: Adaptation with selected pseudo-labels (~7.9 hrs)
By default, run each command below as-is — the Lower bound, Warm-start, and AdvDA variants already load from the precomputed `data/stepII_constructed_datasets/mb24/aug/`. The Upper bound variants use ground-truth labels directly and don't depend on Step II's output at all.

Run each variant to reproduce the corresponding point in Figure 6:

| Method | Command | Approx. runtime |
|---|---|---|
| Lower bound (ResNet) | `python StepIII/Images/Lower_bound/train_mb24+_aug.py` | 8.5 min |
| Lower bound (GIN) | `python StepIII/CFGs/Lower_bound/train_mb24+_aug.py` | 57.4 min |
| Warm-start ResNet-50 | `python StepIII/Images/Warm_start/train_mb24+_aug.py` | 11.8 min |
| Warm-start GIN | `python StepIII/CFGs/Warm_start/train_mb24+_aug.py` | 72.6 min |
| AdvDA + CNN | `python StepIII/Images/AdvDA/train_mb24+_aug.py` | 20.7 min |
| AdvDA + GIN | `python StepIII/CFGs/AdvDA/train_mb24+_aug.py` | 232.6 min |
| Upper bound (ResNet) | `python StepIII/Images/Upper_bound/train_mb24+_aug.py` | 13.4 min |
| Upper bound (GIN) | `python StepIII/CFGs/Upper_bound/train_mb24+_aug.py` | 85.8 min |

<details>
<summary><strong>Optional: edit the data-loading path to use your own Step II output</strong></summary>

If you generated your own constructed dataset in Step II above, first edit the data-loading path near the top of the script you want to run, from the default `data/stepII_constructed_datasets/mb24/aug/...` to `results/stepII_constructed_datasets_scratch/mb24/aug/...`.

</details>

**Total estimated runtime: ~9 hours** (Step I + Step II + all Step III variants above).

## Expected results

### Step I (Figure 3, Aug testing)

Look at the last printed epoch block (`Epoch: 30`) and compare its values against:

| Printed metric | Expected value |
|---|---|
| `Acc test target:` | 80.8 |
| `Macro F1 test target:` | 77.7 |

> [!NOTE]
> These values come from our precomputed model at `data/stepI_trained_models/mb24/aug/` (the one Step II loads by default). Running `python StepI/evaluate_pretrained_aug.py` confirms this deterministically: it reproduces Macro F1 almost exactly and accuracy within about a point, with no training involved. Training Step I from scratch here (`--epochs 30`) is instead a faster, scaled-down experiment to verify the training pipeline itself works, not a guaranteed exact match to that specific model: Step I trains a generator/classifier against a domain discriminator in a minimax game (see `StepI/model.py`), so from-scratch results fluctuate run-to-run and epoch-to-epoch rather than converging monotonically. Values within about **±2 points** of the expected values above count as a successful reproduction; we compare at the last epoch specifically, to keep the comparison methodologically sound.

### Step II (Table 6, Aug testing)

Pseudo-label accuracy under the three settings:

- **(1) Original (Step I only)**: 80.74% accuracy
- **(2) Confidence filtering only**: 87.55% accuracy
- **(3) Confidence + outlier detection**, by method:

| Method | Accuracy | Coverage | ACS (Avg) |
|---|---|---|---|
| Local Outlier Factor (LOF) | 89.07 | 61.9 | 75.5 |
| GMM | 88.85 | 59.2 | 74.0 |
| One-Class SVM | 88.82 | 59.1 | 74.0 |
| Mahalanobis | 88.94 | 57.1 | 73.0 |
| Isolation Forest | 88.47 | 59.1 | 73.8 |

### Step III (Figure 6, Aug testing)

| Method | Accuracy | F1 |
|---|---|---|
| Lower bound (ResNet) | 75.7 | 72.1 |
| Lower bound (GIN) | 71.3 | 70.8 |
| Warm-start ResNet-50 | 80.0 | 76.9 |
| Warm-start GIN | 77.2 | 75.1 |
| AdvDA + CNN | 81.8 | 77.9 |
| AdvDA + GIN | 81.2 | 79.4 |
| Upper bound (ResNet) | 86.4 | 82.1 |
| Upper bound (GIN) | 84.2 | 82.5 |

> [!NOTE]
> Results across runs may vary slightly due to stochastic training.
