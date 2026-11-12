# Results

Segmentation numbers used by the benchmark live in [`semantic_segmentation/semsegbench/Config-Paths.csv`](../semantic_segmentation/semsegbench/Config-Paths.csv). Columns are checkpoints (76). Rows are metrics. `retrieve_existing` in `semsegbench/evals.py` reads this file. Do not treat `attack_results.json` as the paper table; those values are a separate log and do not match this file.

Derived tables, rebuilt only from that CSV:

| File | What a row is |
| --- | --- |
| [`results/model_provenance.csv`](../results/model_provenance.csv) | One checkpoint: config, origin, training fields |
| [`results/semseg_model_summary.csv`](../results/semseg_model_summary.csv) | One checkpoint: clean mIoU, ReM, GAM, parameter count |
| [`results/semseg_evaluations.csv`](../results/semseg_evaluations.csv) | One recorded evaluation |
| [`results/semseg_missing_settings.csv`](../results/semseg_missing_settings.csv) | Paper slot with an empty cell |
| [`results/figures/`](../results/figures) | Clean mIoU against ReM and GAM |

`model_id` is the same string as in the provenance table.

ADE20K InternImage-B and InternImage-L had each other's parameter counts in the source table (B was 255,600,606 and L was 128,387,852). Those two cells now match the Cityscapes and VOC counts for the same backbones: B is 128,387,852 and L is 255,600,606. No other cell in `Config-Paths.csv` was edited.

## What the paper counts as an evaluation

Semantic segmentation: 76 checkpoints × 27 settings = 2,052.

The 27 settings are clean, 15 corruptions, PGD / SegPGD / CosPGD under `linf` and `l2`, the ACDC aggregate, and ACDC fog, snow, rain, and night. Mean, GAM, and NARM rows in the source CSV are summaries, not extra runs.

1,734 of those cells have a value. 318 are empty. The empty cells are listed in `semseg_missing_settings.csv`. They are mostly attacks that were not run, and ACDC on ADE20K and PASCAL VOC (ACDC is only filled for the 32 Cityscapes checkpoints). Empty cells are not filled in.

Object detection: the paper reports 61 detectors and 4,087 evaluations (61 × 67). No per-detector result file is in this repository, so those 4,087 rows are not reconstructed. The manuscript's detector list is [`results/detection_models_manuscript.csv`](../results/detection_models_manuscript.csv).

## Columns in `semseg_evaluations.csv`

`evaluation_id` is `model_id` plus the setting. `aAcc`, `mIoU`, and `mAcc` are copied from the source CSV.

Attack budgets are not stored per cell. Appendix D and `retrieve_existing` define the stored attack columns as 20 iterations, non-targeted: `linf` uses epsilon 8/255 and alpha 0.01; `l2` uses epsilon 64 and alpha 0.1. Corruption columns are severity 3. The 15 corruption names are gaussian noise, shot noise, impulse noise, defocus blur, glass blur, motion blur, zoom blur, snow, frost, fog, brightness, contrast, elastic transform, pixelate, and jpeg compression. `h265` is not part of the segmentation benchmark.

## ReM and GAM

Both use mIoU.

- `ReM_linf` / `ReM_l2`: minimum over PGD, SegPGD, and CosPGD at that norm. `*_attacks_present` is how many of the three attacks have a value. The stored `mIoU_narm_*` cells match this minimum wherever they are filled. A ReM from fewer than three attacks is partial.
- `GAM`: minimum over the 15 corruption mIoUs. `relative_drop_ReM_linf` and `relative_drop_GAM` are `(clean - score) / clean`.

For SegFormer MiT-B0, B1, B2, and B3 on Cityscapes at 512×1024, the stored `mIoU_2d_gam` cell equals the 1024×1024 checkpoint's GAM, not the minimum of that row's corruption scores. `GAM` in the summary is the minimum of the 15 values in the same row. The stored cell is left unchanged in `stored_2d_gam`.

## Rebuild

From the repository root, with the standard library only:

```bash
python analysis/reproduce_results.py
```

This rewrites the CSVs under `results/`, the three SVGs under `results/figures/`, and `semantic_segmentation/SUPPORTED_MODELS.md`. It does not run models.
