# LFreeDA: Label-Free Drift Adaptation in Windows Malware Detection

This repository contains the code and instructions for all the experiments in our paper.


## 1. Big-15 
### Step I: Pseudo-label Generation
We provide the code for producing the performance metrics of Step I of LfreeDA on the Big-15 adaptation tasks, as presented in **Table 1** of the paper.

- [Task Cluster 0](StepI/train_big15_c0_task.py): ```python StepI/train_big15_c0_task.py```
- [Task Cluster 1](StepI/train_big15_c1_task.py): ```python StepI/train_big15_c1_task.py```
- [Task Cluster 2](StepI/train_big15_c2_task.py): ```python StepI/train_big15_c2_task.py```


### Step II: Pseudo-label Selection
The Step II notebooks compute ACS scores for each outlier‐detection method with confidence filtering and report pseudo-label accuracy for three settings—without Step II, with confidence filtering only, and with confidence filtering plus outlier detection—matching the results shown in **Figure 4** of the paper. 

- [Task Cluster 0](StepII/big15/big15_c0.ipynb): *StepII/big15/big15_c0.ipynb*
- [Task Cluster 1](StepII/big15/big15_c1.ipynb): *StepII/big15/big15_c1.ipynb*
- [Task Cluster 2](StepII/big15/big15_c2.ipynb): *StepII/big15/big15_c2.ipynb*

### Step III: Adaptation with Selected Pseudolabels
We provide the implementation of different DA methods in Step III of LFreeDA on the Big-15 adaptation tasks, producing performance metrics reported in **Table 2** of the paper. 


- Task Cluster 0:
  - [Warm-start ResNet-50](StepIII/Images/Warm_start/train_big15_c0.py): ```python StepIII/Images/Warm_start/train_big15_c0.py```  
  - [Warm-start GIN](StepIII/CFGs/Warm_start/train_big15_c0.py): ```python StepIII/CFGs/Warm_start/train_big15_c0.py```  
  - [AdvDA + GIN](StepIII/CFGs/AdvDA/train_big15_c0.py): ```python StepIII/CFGs/AdvDA/train_big15_c0.py```  
  - [AdvDA + CNN](StepIII/Images/AdvDA/train_big15_c0.py): ```python StepIII/Images/AdvDA/train_big15_c0.py```

**Repeat the same directory structure (replace `c0` with `c1` or `c2`) for Task Cluster 1 and Task Cluster 2.**


## 2. MB-24
### Step I: Pseudo-label Generation


We provide the code for producing the performance metrics of Step I of LfreeDA on the MB-24 adaptation tasks, as presented in **Table 3** of the paper.

- [July task](StepI/train_mb24_july_task.py): ```python StepI/train_mb24_july_task.py```
- [Aug task](StepI/train_mb24_aug_task.py): ```python StepI/train_mb24_aug_task.py```


### Step II: Pseudo-label Selection


The Step II notebooks compute ACS scores for each outlier‐detection method with confidence filtering and report pseudo-label accuracy for three settings—without Step II, with confidence filtering only, and with confidence filtering plus outlier detection—matching the results shown in **Figure 6** of the paper. 

- [July task](StepII/mb24/mb24_july_task.ipynb): *StepII/mb24/mb24_july_task.ipynb*
- [Aug task:](StepII/mb24/mb24_aug_task.ipynb): *StepII/mb24/mb24_aug_task.ipynb*


### Step III: Adaptation with Selected Pseudolabels

We provide the implementation of different DA methods in Step III of LFreeDA on the MB-24 adaptation tasks, producing performance metrics reported in **Table 4** of the paper. 


- July task:
  - [Warm-start ResNet-50](StepIII/Images/Warm_start/train_mb24_july.py): ```python StepIII/Images/Warm_start/train_mb24_july.py```  
  - [Warm-start GIN](StepIII/CFGs/Warm_start/train_mb24_july.py): ```python StepIII/CFGs/Warm_start/train_mb24_july.py```  
  - [AdvDA + GIN](StepIII/CFGs/AdvDA/train_mb24_july.py): ```python StepIII/CFGs/AdvDA/train_mb24_july.py```  
  - [AdvDA + CNN](StepIII/Images/AdvDA/train_mb24_july.py): ```python StepIII/Images/AdvDA/train_mb24_july.py```

**Repeat the same directory structure (replace `july` with `aug`) for the Aug task.**


## 3. MalwareDrift

### Step I: Pseudo-label Generation


We provide the code for producing the performance metrics of Step I of LfreeDA on the MalwareDrift, as presented in **Table 6(a)** of the paper.

[Target testing](StepI/train_malwaredrift.py): ```python StepI/train_malwaredrift.py```



### Step II: Pseudo-label Selection
The Step II notebooks compute ACS scores for each outlier‐detection method with confidence filtering and report pseudo-label accuracy for three settings—without Step II, with confidence filtering only, and with confidence filtering plus outlier detection—matching the results shown in **Figure 11** of the paper. 

[Target testing](StepII/malwaredrift/malwaredrift.ipynb): *StepII/malwaredrift/malwaredrift.ipynb*

Note that we also report the initial accuracy after Step I and the pseudo-label accuracy after Step II, grouped by predicted class clabels. The results are reported in **Figure 7** in the paper.

### Step III: Adaptation with Selected Pseudolabels

We provide the implementation of different DA methods in Step III of LFreeDA on the MalwareDrift task, producing performance metrics reported in **Table 6 (b)** of the paper. 


- Target testing:
  - [Warm-start ResNet-50](StepIII/Images/Warm_start/train_malwaredrift.py): ```python StepIII/Images/Warm_start/train_malwaredrift.py```  
  - [Warm-start GIN](StepIII/CFGs/Warm_start/train_malwaredrift.py): ```python StepIII/CFGs/Warm_start/train_malwaredrift.py```  
  - [AdvDA + GIN](StepIII/CFGs/AdvDA/train_malwaredrift.py): ```python StepIII/CFGs/AdvDA/train_malwaredrift.py```  
  - [AdvDA + CNN](StepIII/Images/AdvDA/train_malwaredrift.py): ``` python StepIII/Images/AdvDA/train_malwaredrift.py```





## 4. Data 
Due to size restrictions (~175 GB), the entire `data/` folder cannot be included in this repository. It contains all the data needed (features, labels and pretrained models) to reproduce our experiments and will be released upon paper acceptance.

```plaintext
data/
├── graph_features/
├── image_features/
├── labels/
├── stepII_constructed_datasets/
└── stepI_trained_models/
```



