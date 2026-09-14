# Data Verification: mb24 (labels / image_features / graph_features)

Date: 2026-09-13

## Table 1: Internal consistency within `lfreeda` — labels ↔ image_features ↔ graph_features (mb24)

Malware SHA-256 hashes compared across:
- `labels/mb24/<Month>/*.csv`
- `image_features/mb24/<month>` (excluding `aug_obf` and `sep_obf`)
- `graph_features/mb24/<Month>`

| Month | CSV | Image | Graph | Status |
|---|---|---|---|---|
| April | 1080 | 1080 | 1080 | ✅ ALL MATCH |
| **Aug** | 1613 | 1613 | **1487** | ⚠️ MISMATCH (126 missing from graph) |
| July | 1618 | 1618 | 1618 | ✅ ALL MATCH |
| March | 1505 | 1505 | 1505 | ✅ ALL MATCH |
| May | 1496 | 1496 | 1496 | ✅ ALL MATCH |
| Sep | 1337 | 1337 | 1337 | ✅ ALL MATCH |
| Dec | 1302 | 1302 | 1302 | ✅ ALL MATCH |
| Nov | 1210 | 1210 | 1210 | ✅ ALL MATCH |
| Oct | 1444 | 1444 | 1444 | ✅ ALL MATCH |

### Aug discrepancy detail
126 hashes present in the labels CSV and in `image_features` are missing from `graph_features/mb24/Aug`:
- **122 of 126** are obfuscated samples (also found in `image_features/mb24/aug_obf`) — plausibly explained by CFG/graph extraction failing or being skipped for obfuscated binaries.
- **4 of 126** are normal (non-obfuscated) samples with no explanation found — a genuine, unexplained data gap.
- This gap is identical in both the `lfreeda` and `ndss_ae` copies of `graph_features/mb24/Aug` (byte-for-byte identical directories), so it is baked into the source data, not a sync/copy issue.
- Separately, `graph_features/mb24/Aug/0/cfg_embeddings` has 188 hashes with 4 `.npz` files each instead of the expected 2 (duplicated pairs), while `Aug/1/cfg_embeddings` is clean (all hashes have exactly 2 files, no overlap between label folders 0 and 1).

## Table 2: `lfreeda` vs `ndss_ae` — do the two repo copies match?

| Data type | lfreeda scope | ndss_ae scope | Comparison |
|---|---|---|---|
| `labels/mb24` | 9 months | 6 months (`Datasets/mb24`: April, Aug, July, March, May, Sep) | ✅ Identical (byte-for-byte CSV content) for the 6 shared months |
| `image_features/mb24` | 9 months + `aug_obf`/`sep_obf` | **does not exist anywhere in ndss_ae** | ❌ No ndss_ae data to compare — mb24 image data exists only in lfreeda |
| `image_features/benign_source` | full set (8211 files, 4.5G) | full set (8211 files, 4.5G) | ✅ Fully identical, byte-for-byte |
| `image_features/benign_target` | full set (9154 files, 6.9G) | full set (9154 files, 6.9G) | ✅ Fully identical, byte-for-byte |
| `graph_features/mb24` | 9 months | 6 months (`additional_data/graph_features/mb24`) | ✅ Identical (byte-for-byte, incl. the Aug 126-hash gap reproduced identically) for the 6 shared months |
| `graph_features/benign_source` | full set + extra `clean.py` script | full set | ✅ Content identical — `dataset1` filenames differ (lfreeda: `<idx>_graph_goodware.npz`; ndss_ae: `<idx>_graph_<extra_id>_goodware.npz`) but file content (md5) matches 1:1 for all 6000 files; `dataset2`/`dataset3`/`dataset4` are byte-for-byte identical including filenames |
| `graph_features/benign_target` | full set (11674 files, 16G) | full set (11674 files, 16G) | ✅ Fully identical, byte-for-byte |

## Bottom line

- Within `lfreeda`, every month's labels/image/graph agree perfectly except Aug's `graph_features`, which is missing 126 samples (122 explained by obfuscation, 4 unexplained).
- Across the two repo copies (`lfreeda` vs `ndss_ae`), everything that exists in both matches exactly. `ndss_ae` is missing Dec/Nov/Oct entirely (for `labels` and `graph_features`) and lacks the `image_features/mb24` folder altogether.
