# Results

Segmentation scores are in [`semantic_segmentation/semsegbench/Config-Paths.csv`](../semantic_segmentation/semsegbench/Config-Paths.csv): 76 checkpoints, one column each. `retrieve_existing` in `semsegbench/evals.py` reads this file.

`python analysis/reproduce_results.py` rebuilds the tables below from that file, from the supplementary detection tables, and from `object_detection/configs/author_trained/`.

| File | What a row is |
| --- | --- |
| [`results/model_provenance.csv`](../results/model_provenance.csv) | One checkpoint: config, origin, training settings |
| [`results/evaluation_manifest.csv`](../results/evaluation_manifest.csv) | Stored evaluations, with the model, setting, and score |
| [`results/semseg_model_summary.csv`](../results/semseg_model_summary.csv) | One segmentation checkpoint: clean mIoU, ReM, GAM, parameter count |
| [`results/semseg_evaluations.csv`](../results/semseg_evaluations.csv) | One segmentation evaluation |
| [`results/figures/`](../results/figures) | Clean score against ReM and GAM |
| [`results/detection_evaluations.csv`](../results/detection_evaluations.csv) | One MS-COCO detection evaluation |
| [`results/detection_model_summary.csv`](../results/detection_model_summary.csv) | One detector: clean mAP, ReM, GAM, parameter count |
| [`results/detection_voc_evaluations.csv`](../results/detection_voc_evaluations.csv) | PASCAL VOC detection results, Faster R-CNN and RetinaNet |
| [`results/supplementary/detection/method_with_time_stamp.csv`](../results/supplementary/detection/method_with_time_stamp.csv) | MS-COCO source table, 61 detectors |
| [`results/supplementary/detection/all_results_mscoco.csv`](../results/supplementary/detection/all_results_mscoco.csv) | mAP, mAP50, mAP75, and small, medium, and large boxes |
| [`results/supplementary/detection/iterative_attack_mscoco_better_view.csv`](../results/supplementary/detection/iterative_attack_mscoco_better_view.csv) | PGD and BIM over iterations |
| [`results/supplementary/detection/pascal_results.csv`](../results/supplementary/detection/pascal_results.csv) | PASCAL VOC source table |

`model_id` matches the provenance table.

## Reported settings

Semantic segmentation: 76 checkpoints × 27 settings = 2,052. The settings are clean, 15 corruptions, PGD, SegPGD, and CosPGD under `linf` and `l2`, the ACDC aggregate, and ACDC fog, snow, rain, and night. Mean, GAM, and NARM are summaries.

MS-COCO detection: 61 detectors × 67 settings = 4,087. The settings are clean (`iid`), FGSM, PGD and BIM at iterations 1 through 20, 15 2D corruptions, 8 3D corruptions, and `h265_crf` and `h265_abr`. `gam`, `rem_20`, `mean_2d`, and `mean_3d` are summaries.

Stored scores are in [`results/evaluation_manifest.csv`](../results/evaluation_manifest.csv). PASCAL VOC detection is in [`results/detection_voc_evaluations.csv`](../results/detection_voc_evaluations.csv).

## Scores

`evaluation_id` is `model_id` plus the setting. Segmentation rows carry `aAcc`, `mIoU`, and `mAcc`.

Attacks use 20 iterations and are non-targeted. `linf` uses epsilon 8/255 and alpha 0.01. `l2` uses epsilon 64 and alpha 0.1. Corruptions are severity 3: gaussian noise, shot noise, impulse noise, defocus blur, glass blur, motion blur, zoom blur, snow, frost, fog, brightness, contrast, elastic transform, pixelate, and jpeg compression.

## ReM and GAM

Segmentation ReM and GAM use mIoU.

- `ReM_linf` and `ReM_l2` are the minimum over PGD, SegPGD, and CosPGD at that norm.
- GAM is the minimum over the 15 corruption mIoUs.
- `relative_drop_ReM_linf` and `relative_drop_GAM` are `(clean - score) / clean`.

Detection ReM is the minimum mAP of PGD and BIM at iteration 20, non-targeted `linf`, epsilon 8/255, alpha 0.01. Detection GAM is the minimum mAP over the 15 2D corruptions and the 8 3D corruptions.

## Rebuild

From the repository root:

```bash
python analysis/reproduce_results.py
```

The script uses the standard library. It rewrites the CSVs under `results/`, the SVGs under `results/figures/`, and `semantic_segmentation/SUPPORTED_MODELS.md`.
