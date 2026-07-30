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
| `heavy_chain` | Heavy-chain id |
| `light_chain` | Light-chain id |
| `antigen_chain` | Antigen chain ID(s). May be more than one chain (e.g. `AB`). |
| `heavy_seq` |  `heavy_chain`. |
| `light_seq` |  `light_chain`. |
| `antigen_seq` | complete antigen sequence. Multiple chains are joined with `:`. |


## Split sizes

The row memberships match the paper's supplementary splits. The sequence and chain-role fields below
are the corrected metadata; they are not the historical inputs stored in the released checkpoints.

| Dataset | Train | Val | Test |
|---|---:|---:|---:|
| Paragraph (expanded) | 624 | 214 | 216 |
| PECAN | 195 | 101 | 152 |
| AACDB | 3994 | 674 | 546 |
| MIPE | 528 (`train_val`) | — | 63 |

