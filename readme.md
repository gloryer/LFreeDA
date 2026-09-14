# LFreeDA: Label-Free Drift Adaptation in Windows Malware Detection

The official code for "LFreeDA: Label-Free Drift Adaptation in Windows Malware Detection", accepted by ACSAC 2026.

## Hardware Specifications

We have successfully run the code with the following hardware:

- CPU: `<TODO: fill in>`
- GPU: NVIDIA RTX 4090 (24GB)
- Memory: `<TODO: fill in>`

Additionally, we recommend at least 200 GB of available disk space to store the data.

## Installation

Please follow these steps to set up the environment:

1. Download the data from Zenodo and place it under the current directory.
2. Run `tar -xzvf data.tar.gz` to unpack the compressed file, and do not change the name of the extracted folder (the name should be `data/`).
3. Run `pip install -r requirements.txt` to install the required packages.

## Artifact (Scaled-Down, For Artifact Evaluation)

Reproducing our full evaluation requires running the Step I + Step II + Step III pipeline across all five rolling adaptation tasks (July→Aug, Aug→Sep, Sep→Oct, Oct→Nov, Nov→Dec) on the MB-24+ dataset. On a single GPU, one task takes approximately 9 hours end-to-end, so a full reproduction (~45 GPU-hours) is not practical for artifact evaluation.

For artifact evaluation, we instead provide a **scaled-down, fully reproducible demonstration on a single adaptation task (July → August)**, requiring ~9 hours. The artifact is available in the [`artifact/`](artifact/) directory — see [`artifact/ARTIFACT.md`](artifact/ARTIFACT.md) for instructions, expected results, and environment details.

The instructions below cover the **full 5-task pipeline (all months, not required for artifact evaluation)** and are provided for future research.

## Full Reproduction on MB-24+ (All Five Tasks — Not Required for Artifact Evaluation)
### Step I: Pseudo-label Generation

We provide the code for producing the performance metrics of Step I of LFreeDA on the MB-24+ adaptation tasks, as presented in **Figure 3** of the paper. Step I trains a model on labeled source data and unlabeled target data; applying this model to the target training set at inference is what produces the initial (noisy) pseudo-labels that Step II then filters.

- [Aug testing](StepI/train_mb24+_aug.py): ```python StepI/train_mb24+_aug.py```
- [Sep testing](StepI/train_mb24+_sep.py): ```python StepI/train_mb24+_sep.py```
- [Oct testing](StepI/train_mb24+_oct.py): ```python StepI/train_mb24+_oct.py```
- [Nov testing](StepI/train_mb24+_nov.py): ```python StepI/train_mb24+_nov.py```
- [Dec testing](StepI/train_mb24+_dec.py): ```python StepI/train_mb24+_dec.py```

(Optional) Uncomment the save lines at the end of the script if you would like to save your own trained models to `results/stepI_trained_models_scratch/mb24/<month>/`.

### Step II: Pseudo-label Selection
Step II selects high-quality pseudo-labels from Step I's predictions, based on confidence filtering and outlier detection. We propose an Accuracy-Coverage Score (ACS) to evaluate five different outlier detection methods. The notebooks reproduce Table 6 (accuracy, coverage, and ACS for five outlier detection methods) and compute pseudo-label accuracy under two additional settings described in the paper (original / confidence filtering only). Each notebook reports these results for its own adaptation task; Figure 5 in the paper shows the average across all five tasks.

By default, the notebooks load the trained Step I models directly from `data/stepI_trained_models/mb24/<month>/` — you can just run all cells. We've kept the cell outputs saved in the notebooks so you can compare against the expected results.


- [Aug testing](StepII/mb24+/mb24+_aug.ipynb): ```jupyter notebook StepII/mb24+/mb24+_aug.ipynb```
- [Sep testing](StepII/mb24+/mb24+_sep.ipynb): ```jupyter notebook StepII/mb24+/mb24+_sep.ipynb```
- [Oct testing](StepII/mb24+/mb24+_oct.ipynb): ```jupyter notebook StepII/mb24+/mb24+_oct.ipynb```
- [Nov testing](StepII/mb24+/mb24+_nov.ipynb): ```jupyter notebook StepII/mb24+/mb24+_nov.ipynb```
- [Dec testing](StepII/mb24+/mb24+_dec.ipynb): ```jupyter notebook StepII/mb24+/mb24+_dec.ipynb```

(Optional) To evaluate a different Step I model, point the model-loading cell at `results/stepI_trained_models_scratch/mb24/<month>/{generator,classifier}`.

(Optional) To also save the constructed datasets, uncomment the save block in the last cell — output goes to `results/stepII_constructed_datasets_scratch/mb24/<month>/`.

### Step III: Adaptation with Selected Pseudo-labels

We provide the implementation of different DA methods in Step III of LFreeDA on the MB-24+ adaptation tasks, producing performance metrics reported in **Figure 6** of the paper. Step III trains on the pseudo-labeled target data that Step II selected, comparing several domain adaptation strategies against lower- and upper-bound baselines.

By default, you can run the commands below as-is — the Lower bound, Warm-start, and AdvDA variants already load from the precomputed `data/stepII_constructed_datasets/mb24/<month>/`. The Upper bound variants are the exception: they train directly on ground-truth target labels from `data/labels/` rather than Step II's pseudo-labels, so they don't depend on Step II's output at all.

- August testing:
  - [Lower bound (ResNet)](StepIII/Images/Lower_bound/train_mb24+_aug.py): ```python StepIII/Images/Lower_bound/train_mb24+_aug.py```
  - [Lower bound (GIN)](StepIII/CFGs/Lower_bound/train_mb24+_aug.py): ```python StepIII/CFGs/Lower_bound/train_mb24+_aug.py```
  - [Upper bound (ResNet)](StepIII/Images/Upper_bound/train_mb24+_aug.py): ```python StepIII/Images/Upper_bound/train_mb24+_aug.py```
  - [Upper bound (GIN)](StepIII/CFGs/Upper_bound/train_mb24+_aug.py): ```python StepIII/CFGs/Upper_bound/train_mb24+_aug.py```
  - [Warm-start ResNet-50](StepIII/Images/Warm_start/train_mb24+_aug.py): ```python StepIII/Images/Warm_start/train_mb24+_aug.py```
  - [Warm-start GIN](StepIII/CFGs/Warm_start/train_mb24+_aug.py): ```python StepIII/CFGs/Warm_start/train_mb24+_aug.py```
  - [AdvDA + GIN](StepIII/CFGs/AdvDA/train_mb24+_aug.py): ```python StepIII/CFGs/AdvDA/train_mb24+_aug.py```
  - [AdvDA + CNN](StepIII/Images/AdvDA/train_mb24+_aug.py): ```python StepIII/Images/AdvDA/train_mb24+_aug.py```

**Repeat the same directory structure (replace `aug` with `sep`, `oct`, `nov`, `dec`) for the Sep, Oct, Nov, Dec testing.**

## Data
Due to size restrictions (~140 GB), the full `data/` folder cannot be included in this repository. It contains all the data needed (features, labels, and trained models) to reproduce our experiments and will be released upon paper acceptance.

```plaintext
data/
├── graph_features/
├── image_features/
├── labels/
├── stepII_constructed_datasets/
└── stepI_trained_models/
```

`stepI_trained_models/` and `stepII_constructed_datasets/` contain our own precomputed Step I and Step II outputs, used by default.

## MB-24+ Original Binaries

Original malware binaries can be downloaded from [MalwareBazaar](https://bazaar.abuse.ch/) using the provided SHA-256 hashes. SHA-256 hashes are listed by month in [`data/labels/mb24`](data/labels/mb24).

Note: Due to copyright restrictions, we are unable to share the Windows benign samples.

> [!IMPORTANT]
> For how to extract graph features from binaries, please see [malware-detection-concept-drift](https://github.com/gloryer/malware-detection-concept-drift) for details.

## Citation

If you use the code or dataset used here, please cite our paper:

```bibtex
@inproceedings{lfreeda2026,
  title     = {LFreeDA: Label-Free Drift Adaptation for Windows Malware Detection},
  author    = {Adrian and Shuai Li and Elisa Bertino},
  booktitle = {Annual Computer Security Applications Conference (ACSAC)},
  year      = {2026}
}
```

