# LFreeDA: Label-Free Drift Adaptation in Windows Malware Detection

This repository contains the code and instructions for running the experiments in our paper.




## MB-24+
### Step I: Pseudo-label Generation


We provide the code for producing the performance metrics of Step I of LfreeDA on the MB-24+ adaptation tasks, as presented in **Figure 4** of the paper.

- [Aug testing](StepI/train_mb24+_aug.py): ```python StepI/train_mb24+_aug.py```
- [Sep testing](StepI/train_mb24+_sep.py): ```python StepI/train_mb24+_sep.py```
- [Oct testing](StepI/train_mb24+_oct.py): ```python StepI/train_mb24+_oct.py```
- [Nov testing](StepI/train_mb24+_nov.py): ```python StepI/train_mb24+_nov.py```
- [Dec testing](StepI/train_mb24+_dec.py): ```python StepI/train_mb24+_dec.py```

### Step II: Pseudo-label Selection


The Step II notebooks compute ACS scores for each outlier-detection method with confidence filtering. We report pseudo-label accuracy for three settings: (1) without Step II, (2) with confidence filtering only, and (3) with confidence filtering plus outlier detection (Step II). These results correspond to **Figure 6** in the paper.


- [Aug testing](StepII/mb24+/mb24+_aug.ipynb): ```jupyter notebook StepII/mb24+/mb24+_aug.ipynb```
- [Sep testing](StepII/mb24+/mb24+_sep.ipynb): ```jupyter notebook StepII/mb24+/mb24+_sep.ipynb```
- [Oct testing](StepII/mb24+/mb24+_oct.ipynb): ```jupyter notebook StepII/mb24+/mb24+_oct.ipynb```
- [Nov testing](StepII/mb24+/mb24+_nov.ipynb): ```jupyter notebook StepII/mb24+/mb24+_nov.ipynb```
- [Dec testing](StepII/mb24+/mb24+_dec.ipynb): ```jupyter notebook StepII/mb24+/mb24+_dec.ipynb```


### Step III: Adaptation with Selected Pseudolabels

We provide the implementation of different DA methods in Step III of LFreeDA on the MB-24+ adaptation tasks, producing performance metrics reported in **Figure 7** of the paper.


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





## 4. Data 
Due to size restrictions (~200 GB), the full`data/` folder cannot be included in this repository. It contains all the data needed (features, labels, and trained models) to reproduce our experiments and will be released upon paper acceptance.

```plaintext
data/
├── graph_features/
├── image_features/
├── labels/
├── stepII_constructed_datasets/
└── stepI_trained_models/
```



