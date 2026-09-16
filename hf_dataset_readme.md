---
license: cc0-1.0
---

# LFreeDA Dataset (MB-24+)

Data accompanying **"LFreeDA: Label-Free Drift Adaptation for Windows Malware Detection"** (ACSAC 2026).

This dataset provides the precomputed features, labels, and intermediate artifacts needed to reproduce the paper's Step I–III pipeline on the MB-24+ malware corpus, spanning five rolling monthly adaptation tasks (July→Aug, Aug→Sep, Sep→Oct, Oct→Nov, Nov→Dec 2024).

Code: https://github.com/gloryer/LFreeDA

## Contents

- `graph_features/` — control-flow-graph (CFG) embeddings and adjacency matrices per binary, organized by month and malware/benign source.
- `image_features/` — CFG-derived image representations per binary, organized by month and malware/benign source.
- `labels/` — per-month malware labels (SHA-256 hash, malware family, and binary attack label).
- `stepI_trained_models/` — pretrained Step I generator/classifier weights for all five adaptation tasks.
- `stepII_constructed_datasets/` — precomputed Step II pseudo-label-filtered training/test sets for all five adaptation tasks, used directly by the Step III Warm-start/AdvDA scripts.

See the code repository's `readme.md` and `artifact/ARTIFACT.md` for exact usage instructions and expected results.

## License

Released under **CC0-1.0** (public domain dedication).

## Citation

If you use this dataset, please cite our paper:

```bibtex
@inproceedings{lfreeda2026,
  title     = {LFreeDA: Label-Free Drift Adaptation for Windows Malware Detection},
  author    = {Adrian and Shuai Li and Elisa Bertino},
  booktitle = {Annual Computer Security Applications Conference (ACSAC)},
  year      = {2026}
}
```
