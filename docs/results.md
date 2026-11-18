# Results

The release records 5,780 evaluations: 1,734 semantic segmentation evaluations on 76 checkpoints, and 4,046 object-detection evaluations on 63 detectors.

`python analysis/reproduce_results.py` rebuilds the tables and figures from the stored result files and from `object_detection/configs/author_trained/`. It does not run models, attacks, or training.

## Result files

| File | Contents |
| --- | --- |
| [`results/evaluation_manifest.csv`](../results/evaluation_manifest.csv) | All 5,780 recorded evaluations |
| [`results/model_provenance.csv`](../results/model_provenance.csv) | 76 segmentation checkpoints and 63 detectors |
| [`results/semseg_evaluations.csv`](../results/semseg_evaluations.csv) | The 1,734 segmentation evaluations |
| [`results/semseg_model_summary.csv`](../results/semseg_model_summary.csv) | One segmentation checkpoint: clean mIoU, ReM, GAM, parameter count |
| [`results/detection_evaluations.csv`](../results/detection_evaluations.csv) | MS-COCO rows of the object-detection benchmark |
| [`results/detection_voc_evaluations.csv`](../results/detection_voc_evaluations.csv) | PASCAL VOC rows of the object-detection benchmark |
| [`results/detection_model_summary.csv`](../results/detection_model_summary.csv) | One detector: clean mAP, ReM, GAM, parameter count |
| [`results/figures/`](../results/figures) | Clean score against ReM and GAM |
| [`semantic_segmentation/semsegbench/Config-Paths.csv`](../semantic_segmentation/semsegbench/Config-Paths.csv) | Segmentation source table, 76 checkpoints |
| [`results/supplementary/detection/method_with_time_stamp.csv`](../results/supplementary/detection/method_with_time_stamp.csv) | MS-COCO detection source table |
| [`results/supplementary/detection/pascal_results.csv`](../results/supplementary/detection/pascal_results.csv) | PASCAL VOC detection source table |
| [`results/supplementary/detection/all_results_mscoco.csv`](../results/supplementary/detection/all_results_mscoco.csv) | mAP, mAP50, mAP75, and small, medium, and large boxes |
| [`results/supplementary/detection/iterative_attack_mscoco_better_view.csv`](../results/supplementary/detection/iterative_attack_mscoco_better_view.csv) | PGD and BIM over iterations |

`retrieve_existing` in `semantic_segmentation/semsegbench/evals.py` reads `Config-Paths.csv`. `model_id` matches the provenance table.

## A manifest row

`evaluation_id` is `model_id` plus the setting. Segmentation rows use mIoU (`aAcc` and `mAcc` are also in `semseg_evaluations.csv`). Object-detection rows use mAP.

`result_file` is the source table. `config`, `config_url`, `checkpoint`, and `origin` come from the matching row of `results/model_provenance.csv`. `provenance_file` points at that table. `analysis` is `analysis/reproduce_results.py`, the script that writes the manifest, the summaries, and the figures.

Segmentation settings in the recorded table are clean images, the 15 severity-3 2D corruptions, PGD, SegPGD, and CosPGD under `linf` and `l2`, and ACDC. Attacks use 20 iterations and are non-targeted. `linf` uses epsilon 8/255 and alpha 0.01. `l2` uses epsilon 64 and alpha 0.1. The corruptions are gaussian noise, shot noise, impulse noise, defocus blur, glass blur, motion blur, zoom blur, snow, frost, fog, brightness, contrast, elastic transform, pixelate, and jpeg compression.

Object-detection settings in the recorded table are clean (`iid`), FGSM, PGD, BIM, the 15 2D corruptions, and the 8 3D corruptions. PGD and BIM are non-targeted `linf`, epsilon 8/255, alpha 0.01. The MS-COCO source table also records `h265_crf` and `h265_abr`.

## ReM and GAM

Segmentation ReM and GAM use mIoU.

- `ReM_linf` and `ReM_l2` are the minimum over PGD, SegPGD, and CosPGD at that norm.
- GAM is the minimum over the 15 corruption mIoUs.
- `relative_drop_ReM_linf` and `relative_drop_GAM` are `(clean - score) / clean`.

Detection ReM is the minimum mAP of PGD and BIM at iteration 20, non-targeted `linf`, epsilon 8/255, alpha 0.01. Detection GAM is the minimum mAP over the 15 2D corruptions and the 8 3D corruptions.

`results/figures/ade20k.svg`, `cityscapes.svg`, and `voc.svg` plot clean mIoU against those segmentation scores. `results/figures/coco.svg` plots clean mAP against ReM and GAM for the MS-COCO rows of `detection_model_summary.csv`.

## Rebuild

From the repository root:

```bash
python analysis/reproduce_results.py
```

The script uses the standard library. It rewrites the CSVs under `results/`, the SVGs under `results/figures/`, and `semantic_segmentation/SUPPORTED_MODELS.md`.
