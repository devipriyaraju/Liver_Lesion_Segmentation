# Uploaded experiments

Each complete run directory contains the original trained `final.pt` checkpoint and seven uploaded CSVs: `train_log.csv`, `cases.csv`, `cases_livermask.csv`, `lesions.csv`, `lesions_livermask.csv`, `froc.csv`, and `froc_livermask.csv`.

All ten complete runs cover folds 0 and 1 for their respective model/sampling configurations. The run folder names encode architecture, sampler, and fold; the `cases.csv` values were checked against the folder names.

`unet__standard__fold2` is **incomplete**: it only contains `last.pt` and `train_log.csv`. `last.pt` is preserved under its supplied name and **must not be treated as an evaluated `final.pt` checkpoint**. No case, lesion, FROC, or liver-mask evaluation tables were supplied for fold 2.

Model weights are tracked with Git LFS. These files are large; ensure the Git LFS storage and bandwidth allowances are sufficient before pushing to GitHub.
