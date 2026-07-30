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

| Dataset | Train | Val | Test |
|---|---:|---:|---:|
| Paragraph (expanded) | 624 | 214 | 216 |
| PECAN | 195 | 101 | 152 |
| AACDB | 3994 | 674 | 546 |
| MIPE | 528 (`train_val`) | — | 63 |

