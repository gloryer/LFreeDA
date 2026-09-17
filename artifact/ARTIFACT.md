# LFreeDA Artifact (Scaled-Down)

## Table of Contents

- [Scope](#scope)
- [What this artifact reproduces](#what-this-artifact-reproduces)
- [Environment](#environment)
- [Data required](#data-required)
- [How to run](#how-to-run)
  - [Step I: Pseudo-label generation](#step-i-pseudo-label-generation-122-min)
  - [Step II: Pseudo-label selection](#step-ii-pseudo-label-selection-5-min)
  - [Step III: Adaptation with selected pseudo-labels](#step-iii-adaptation-with-selected-pseudo-labels-28-hrs)
- [Expected results](#expected-results)

## Scope

This artifact provides a **scaled-down, fully reproducible demonstration of LFreeDA on a single adaptation task: July → August 2024**. The paper's full evaluation spans five rolling adaptation tasks (July→Aug, Aug→Sep, Sep→Oct, Oct→Nov, Nov→Dec) at full training epochs, each taking ~6.5 hours end-to-end on a single GPU — ~32.5 GPU-hours total, which isn't practical within a typical artifact evaluation window.

Running this artifact takes you through one representative task, end to end, using reduced training epochs to keep runtime practical. This lets you independently verify that the pipeline works and compare your results against the reference values under [Expected results](#expected-results) below — the goal is a result consistent with the paper within normal run-to-run variance, not an exact match. Code for the full 5-task, full-epoch reproduction is in the main repository, documented in the top-level [readme.md](../readme.md) — provided for future research, but **not** part of this artifact's evaluation scope.

## What this artifact reproduces

Using the reduced-epoch, single-task run described above, this artifact reproduces the following paper results — compare your output against the reference values under [Expected results](#expected-results), allowing for normal run-to-run variance rather than an exact match:

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

### Step I: Pseudo-label generation (~12.2 min)
```bash
python StepI/train_mb24+_aug.py --epochs 30
```
Trains the Step I model and reports test accuracy for the July→Aug task. 

<details>
<summary><strong>Optional: save your own trained model</strong></summary>

The save block at the end of the script is commented out by default. Uncomment it if you'd also like to save the trained model to `results/stepI_trained_models_scratch/mb24/aug/`.

</details>

<details>
<summary><strong>Optional: evaluate the trained model directly (no training, deterministic)</strong></summary>

```bash
python StepI/evaluate_pretrained_aug.py
```
Loads `data/stepI_trained_models/mb24/aug/` (the same model Step II uses by default) and reports its `Acc test target`/`Macro F1 test target` on the July→Aug task's test set, without training anything — useful as a deterministic sanity check against the expected values below, independent of training-run variance.

</details>

### Step II: Pseudo-label selection (~5 min)
```bash
jupyter notebook StepII/mb24+/mb24+_aug.ipynb
```
> [!TIP]
> If you're using VS Code (including over SSH), skip the command above — install the **Jupyter** and **Python** extensions if you haven't, then open the `.ipynb` file directly in the editor.
> Click **"Select Kernel"** (top right of the notebook) → **Python Environments** → pick the `lfreeda` conda env, then **Run All**.

By default, the notebook loads the trained Step I model directly from `data/stepI_trained_models/mb24/aug/`. You can just run all cells.

This reproduces:
- Table 6 (accuracy, coverage, and ACS for five outlier detection methods, July→Aug task)
- pseudo-label accuracy under two additional settings described in the paper (original / confidence filtering only)

The setting Figure 5 in the paper reports as an average across all five tasks, so compare your July→Aug numbers against the reference values under [Expected results](#expected-results) rather than Figure 5 directly.

We've kept the cell outputs saved in the notebook so you can compare against the expected results.

<details>
<summary><strong>Optional: evaluate a different Step I model, or save the constructed datasets</strong></summary>

- To evaluate a different Step I model, point the model-loading cell at `results/stepI_trained_models_scratch/mb24/aug/{generator,classifier}`.
- To also save the constructed datasets, uncomment the save block in the last cell — output goes to `results/stepII_constructed_datasets_scratch/mb24/aug/`.

</details>

### Step III: Adaptation with selected pseudo-labels (~2.8 hrs)
By default, run each command below as-is — the Lower bound, Warm-start, and AdvDA variants already load from the precomputed `data/stepII_constructed_datasets/mb24/aug/`. The Upper bound variants use ground-truth labels directly and don't depend on Step II's output at all.

Run each command to train the step III models and reports test accuracy for the July→Aug task. 

| Method | Command | Approx. runtime (`--epochs 20`) |
|---|---|---|
| Lower bound (ResNet) | `python StepIII/Images/Lower_bound/train_mb24+_aug.py --epochs 20` | 2.4 min |
| Lower bound (GIN) | `python StepIII/CFGs/Lower_bound/train_mb24+_aug.py --epochs 20` | 24.1 min |
| Warm-start ResNet-50 | `python StepIII/Images/Warm_start/train_mb24+_aug.py --epochs 20` | 2.5 min |
| Warm-start GIN | `python StepIII/CFGs/Warm_start/train_mb24+_aug.py --epochs 20` | 30.5 min |
| AdvDA + CNN | `python StepIII/Images/AdvDA/train_mb24+_aug.py --epochs 20` | 3.6 min |
| AdvDA + GIN | `python StepIII/CFGs/AdvDA/train_mb24+_aug.py --epochs 20` | 73.1 min |
| Upper bound (ResNet) | `python StepIII/Images/Upper_bound/train_mb24+_aug.py --epochs 20` | 2.9 min |
| Upper bound (GIN) | `python StepIII/CFGs/Upper_bound/train_mb24+_aug.py --epochs 20` | 28.6 min |


<details>
<summary><strong>Optional: edit the data-loading path to use your own Step II output</strong></summary>

If you generated your own constructed dataset in Step II above, first edit the data-loading path near the top of the script you want to run, from the default `data/stepII_constructed_datasets/mb24/aug/...` to `results/stepII_constructed_datasets_scratch/mb24/aug/...`.

</details>

**Total estimated runtime: ~3.1 hours** (Step I + Step II + all Step III variants above).

## Expected results

### Step I (Figure 3, Aug testing)

Look at the last printed epoch block (`Epoch: 30`) and compare its values against:

| Printed metric | Expected value |
|---|---|
| `Acc test target:` | 80.8 |
| `Macro F1 test target:` | 77.7 |

> [!NOTE]
> - `python StepI/evaluate_pretrained_aug.py` confirms these values deterministically (no training). 
> - Training from scratch (`--epochs 30`) is a faster, scaled-down check that the pipeline works, not a guaranteed exact match to that specific model.
> - Step I trains a generator/classifier against a domain discriminator in a minimax game (see `StepI/model.py`), so from-scratch results fluctuate run-to-run and epoch-to-epoch rather than converging monotonically.
> - Values within about **±2 points** of the expected values above count as a successful reproduction.

### Step II (Table 6, Aug testing)

Pseudo-label accuracy under the three settings:

- **(1) Original (Step I only)**: 81.16% accuracy
- **(2) Confidence filtering only**: 87.51% accuracy
- **(3) Confidence + outlier detection**, by method:

| Method | Accuracy | Coverage | ACS (Avg) |
|---|---|---|---|
| Local Outlier Factor (LOF) | 89.17 | 61.9 | 75.5 |
| GMM | 88.56 | 59.0 | 73.8 |
| One-Class SVM | 88.59 | 58.9 | 73.7 |
| Mahalanobis | 88.74 | 56.8 | 72.8 |
| Isolation Forest | 88.33 | 58.9 | 73.6 |

### Step III (Figure 6, Aug testing)

| Method | Accuracy | F1 | Reproduced Accuracy | Reproduced F1 |
|---|---|---|---|---|
| Lower bound (ResNet) | 75.7 | 72.1 | 76.7 | 73.2 |
| Lower bound (GIN) | 71.3 | 70.8 | 67.9 | 67.5 |
| Warm-start ResNet-50 | 80.0 | 76.9 | 80.6 | 75.2 |
| Warm-start GIN | 77.2 | 75.1 | 80.6 | 77.8 |
| AdvDA + CNN | 81.8 | 77.9 | 82.5 | 78.9 |
| AdvDA + GIN | 81.2 | 79.4 | 81.3 | 79.4 |
| Upper bound (ResNet) | 86.4 | 82.1 | 83.8 | 80.6 |
| Upper bound (GIN) | 84.2 | 82.5 | 82.5 | 80.2 |

> [!NOTE]
> Results across runs may vary slightly due to stochastic training.
