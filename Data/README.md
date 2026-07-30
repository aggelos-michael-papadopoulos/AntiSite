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
| `heavy_chain` | Heavy-chain role used by the model. It is a biological heavy chain except in the documented nonstandard AACDB rows. |
| `light_chain` | Light-chain role used by the model. Every row is non-empty, but the documented AACDB fusion/VL-only rows are not conventional light chains. |
| `antigen_chain` | Antigen chain ID(s). May be more than one chain (e.g. `AB`). |
| `heavy_seq` | Atom-resolved, Fv-sized model input window for `heavy_chain`. |
| `light_seq` | Atom-resolved, Fv-sized model input window for `light_chain`. |
| `antigen_seq` | All observed CA residues for the antigen chain(s), without antibody-style truncation. Multiple chains are joined with `:`. |

Sequences are extracted from the structures with the **same** routine AntiSite uses at
train/inference time (`antisite.eval.labels.extract_sequence`), so they align exactly with the
model's inputs. For normally numbered antibody chains, the historical author-number selection
(`resnum <= 128`) is retained when it contains at least 80 observed residues. For offset-numbered
chains, the first 128 observed CA residues are used instead. This is an Fv-sized model-input heuristic,
not a newly validated domain boundary. Antigens use all observed CA residues. No residue absent from
the atomic coordinates is invented from `SEQRES`.

## Split sizes

The row memberships match the paper's supplementary splits. The sequence and chain-role fields below
are the corrected metadata; they are not the historical inputs stored in the released checkpoints.

| Dataset | Train | Val | Test | Notes |
|---|---:|---:|---:|---|
| Paragraph (expanded) | 624 | 214 | 216 | Primary benchmark (headline model). |
| PECAN | 195 | 101 | 152 | |
| AACDB | 3994 | 674 | 546 | Published memberships preserved; corrected sequences are not cluster-disjoint (see audit). |
| MIPE | 528 (`train_val`) | — | 63 | **5-fold** cross-validation over `train_val`. |

## Notes

- Every one of the 7,307 rows has non-empty `heavy_seq`, `light_seq`, and `antigen_seq` values.
- Five AACDB light chains have only 67–78 atom-resolved residues. They are retained as observed rather
  than padded with residues that have no coordinates.
- Six AACDB rows are structurally nonstandard (fusion, diabody, or no conventional heavy chain). Their
  published split IDs are preserved, but they must not be described as ordinary paired H/L complexes.
- These splits reflect AntiSite's own filtering/feature-extraction pipeline; they are **not** identical
  to ParaSurf's published `training_data/` splits, which come from a different pipeline over the same
  source datasets.

The corrected AACDB strings place 21 CD-HIT-70% clusters across the preserved split boundaries.
Restoring cluster separation requires a new split rather than a sequence-column repair. Existing
released checkpoints and published metrics represent the historical inputs; corrected model results
require rebuilding examples and embeddings and then retraining.
