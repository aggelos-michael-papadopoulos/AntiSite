# AntiSite — Datasets 📁

Train / validation / test splits for the four benchmarks AntiSite is trained and evaluated on.
Each row lists the antibody **heavy** / **light** chain IDs, the **antigen** chain(s), and the
corresponding **sequences**, so the splits are self-contained and reproducible.

```
Data/
├── Paragraph/   train.csv  val.csv  test.csv
├── PECAN/        train.csv  val.csv  test.csv
├── AACDB/        train.csv  val.csv  test.csv
└── MIPE/         train_val.csv  test.csv     # 5-fold CV is run over train_val
```

## Columns

| Column | Description |
|---|---|
| `pdb_code` | Structure identifier. For AACDB it also encodes the chain grouping (e.g. `1FJ1_BAF`), matching the source PDB file. |
| `heavy_chain` | Antibody heavy-chain ID. |
| `light_chain` | Antibody light-chain ID. **Empty** for single-domain antibodies (nanobodies). |
| `antigen_chain` | Antigen chain ID(s). May be more than one chain (e.g. `AB`). |
| `heavy_seq` | Heavy-chain **Fv** sequence (variable domain, PDB resnum ≤ 128). |
| `light_seq` | Light-chain **Fv** sequence (empty for nanobodies). |
| `antigen_seq` | Antigen sequence(s), full chain. Multiple antigen chains are joined with `:`. |

Sequences are extracted from the structures with the **same** routine AntiSite uses at
train/inference time (`antisite.eval.labels.extract_sequence`), so they align exactly with the
model's inputs — antibody chains trimmed to the Fv domain, antigen chains kept full length.

## Split sizes

These are the exact splits AntiSite is trained and evaluated on (taken from the built training
examples), matching the paper's supplementary.

| Dataset | Train | Val | Test | Notes |
|---|---:|---:|---:|---|
| Paragraph (expanded) | 624 | 214 | 216 | Primary benchmark (headline model). |
| PECAN | 195 | 101 | 152 | |
| AACDB | 3994 | 674 | 546 | Cluster-disjoint splits (CD-HIT, 70% identity). |
| MIPE | 528 (`train_val`) | — | 63 | **5-fold** cross-validation over `train_val`. |

## Notes

- **Antibody sequences are Fv only** (resnum ≤ 128); this is the region AntiSite predicts on. Antigen
  sequences are the full chain.
- **Empty `light_seq`** marks a single-domain/nanobody input, or an entry whose light chain could not be
  Fv-extracted; **empty `heavy_seq`** marks a handful of entries that are effectively light-only under
  the Fv-numbering convention. These are kept exactly as the model was trained on them.
- These splits reflect AntiSite's own filtering/feature-extraction pipeline; they are **not** identical
  to ParaSurf's published `training_data/` splits, which come from a different pipeline over the same
  source datasets.
