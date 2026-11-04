# AntiSite — Datasets 📁

Train / validation / test splits for the four original benchmarks, plus the final **SAbDab2
paired-antibody test set (178 complexes)** and **zero-shot VHH test set (140 complexes)**.
Each row lists antibody and antigen chain IDs and their sequences. The two additional test sets
were evaluated using the released Paragraph-expanded AntiSite checkpoint without retraining.

```
Data/
├── Paragraph/    train.csv  val.csv  test.csv
├── PECAN/        train.csv  val.csv  test.csv
├── AACDB/        train.csv  val.csv  test.csv
├── MIPE/         train_val.csv  test.csv     # 5-fold CV is run over train_val
├── SabDab2/      test_final.csv             # final shared 178-complex comparison
└── VHH/          test_final.csv             # 140-complex zero-shot evaluation
```

## Columns

| Column | Description |
|---|---|
| `pdb_code` | Structure identifier. For AACDB it also encodes the chain grouping (e.g. `1FJ1_BAF`), matching the source PDB file. |
| `heavy_chain` | Heavy-chain ID; also used for the VHH domain in nanobodies. |
| `light_chain` | Light-chain ID; empty for VHHs. |
| `antigen_chain` | Antigen chain ID(s). May be more than one chain (e.g. `AB`). |
| `heavy_seq` | Atom-resolved sequence of the selected heavy-chain variable domain or VHH. |
| `light_seq` | Atom-resolved sequence of the selected light-chain variable domain; empty for VHHs. |
| `antigen_seq` | Atom-resolved antigen sequence. Multiple chains are joined with `:` in the declared chain order. |

Antigen fields identify the reference complex and support contact-label construction and
antigen-aware comparison methods. **AntiSite does not use antigen sequence or coordinates as inputs.**

The two `test_final.csv` files also include source identifiers:

| Column | Description |
|---|---|
| `instance` | Original SAbDab2 structure-instance identifier. |
| `orig_heavy_chain`, `orig_light_chain`, `orig_antigen_chain` | Chain IDs in the source structure, before renaming; case-sensitive. |
| `SABDAB_ID` | SAbDab2 antibody identifier. |
| `ab_cluster` | Antibody cluster in the source split. |

In these processed test sets, antibody chains are named `H` and `L`, and the antigen is named `A`.
VHHs have only `H`; their light-chain fields are empty. Use the `orig_*` columns when locating the
corresponding chains in the original structures.

## Split sizes

The four original benchmark memberships are unchanged. Their sequence and chain-role fields contain
the previously corrected metadata and need not match the historical inputs used to train the
released checkpoints. The two new test-only manifests retain the exact antibody sequences from
the saved SAbDab2/VHH evaluation inputs.

| Dataset | Train | Val | Test |
|---|---:|---:|---:|
| Paragraph (expanded) | 624 | 214 | 216 |
| PECAN | 195 | 101 | 152 |
| AACDB | 3994 | 674 | 546 |
| MIPE | 528 (`train_val`) | — | 63 |
| SAbDab2, paired antibodies | — | — | 178 |
| VHH, zero-shot | — | — | 140 |

## Additional test sets

### SAbDab2 paired antibodies

[`SabDab2/test_final.csv`](SabDab2/test_final.csv) contains the **178 complexes used in the main
benchmark comparison**. It is the final common cohort after filtering for paired antibodies and a
contacting protein/peptide antigen, excluding antibodies similar to available training references,
and applying structure-preparation, predictor-input and potential ParaAntiProt-training-overlap
checks. Similarity filtering used at least 85% identity in CDR-H3 or concatenated heavy- or
light-chain CDRs against the Paragraph-expanded and Paraplume references. Cohort membership was
not selected by prediction performance.

### VHH nanobodies

[`VHH/test_final.csv`](VHH/test_final.csv) contains **140 genuine VHH–antigen complexes representing
104 distinct antibodies**, from SAbDab2's single-heavy-domain test split. These are nanobodies, not
paired antibodies with their light chains removed. The evaluation used the paired-antibody-trained
model without VHH training or model selection. The cohort has no shared PDB identifier or exact
CDR-H3 sequence with the AntiSite/ParaSurf training references.

Both sets originate from the [SAbDab2 ML data release](https://doi.org/10.5281/zenodo.22019991):
`ab_split.csv` for paired antibodies and `ab_split_sd.csv` for single domains. The manifests contain
metadata and sequences; source coordinates are available in that release. Training and validation
partitions from SAbDab2 or its VHH subset were not used to fit the released AntiSite model and are
not included in these test-only additions.

