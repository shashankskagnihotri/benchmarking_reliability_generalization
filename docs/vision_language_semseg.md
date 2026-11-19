# RGB-text semantic segmentation extension

This extension evaluates six open-vocabulary RGB-text segmentors on the SemSegBench validation splits. It does not change the published SemSegBench tables in `results/semseg_evaluations.csv` or `results/semseg_model_summary.csv`. New results are written under `results/vision_language_semseg/`.

## Models

| Method | Variant | Backbone | Training data | Checkpoint |
| --- | --- | --- | --- | --- |
| SAN | `san-vit-l14_coco-stuff164k-640x640` | CLIP ViT-L/14 | COCO-Stuff 164k | [san-vit-l14_20230907-a11e098f.pth](https://download.openmmlab.com/mmsegmentation/v0.5/san/san-vit-l14_20230907-a11e098f.pth) |
| CAT-Seg | CAT-Seg (L), `configs/vitl_336.yaml` | CLIP ViT-L/14@336px | COCO-Stuff | [model_large.pth](https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_large.pth) |
| SED | SED (L), `configs/convnextL_768.yaml`, `TEST.FAST_INFERENCE=False` | ConvNeXt-L | COCO-Stuff | [ConvNeXt-L checkpoint](https://drive.google.com/file/d/1zAXE0QXy47n0cVn7j_2cSR85eqxdDGg8/view) |
| FC-CLIP | ResNet50x16 semantic/panoptic head used for semantic evaluation | CLIP ResNet50x16 | COCO panoptic | [fcclip_cocopan_r50x16](https://drive.google.com/file/d/1rJu2UwwtI0VLDl9CyB8Mkgw9qHlnX70W/view) |
| MAFT+ | MAFTP-Large semantic, `maftp_l.pth` | CLIP ConvNeXt-L | COCO-Stuff semantic | [maftp_l.pth](https://drive.google.com/file/d/1EQo5guVuKkSSZj4bv0FQN_4X9h_Rwfe5/view) |
| SAM 3 | original image model with text concepts, not SAM 3.1 | SAM 3 | SA-Co | [facebook/sam3](https://huggingface.co/facebook/sam3) |

None of these checkpoints is trained or fine-tuned on ADE20K, Cityscapes, or ACDC for this extension. Source revisions are in `results/vision_language_semseg/model_manifest.csv`.

SED (L) and SED-fast (L) point at the same published weight file. This extension loads that file with `TEST.FAST_INFERENCE=False`, which is the SED (L) decoder.

## Splits

- ADE20K validation: 2,000 images, 150 classes. Annotation 0 is ignored and the remaining labels are shifted down by one, matching `LoadAnnotations(reduce_zero_label=True)`.
- Cityscapes validation: 500 images, 19 trainIds. Label 255 is ignored.
- ACDC official validation: fog 100, night 106, rain 100, snow 100, 406 images in total. The same 19 Cityscapes trainIds are used. The official unlabeled test split is not evaluated. The directory name `test` in `configs/_base_/datasets/acdc.py` is not this split.

Images are first resized with the SemSegBench keep-ratio scales: ADE20K `(2048, 512)`, Cityscapes and ACDC `(2048, 1024)`. Each model then applies its own differentiable resize and normalization. Predictions are bilinearly resized to the original label resolution before the metric, with `align_corners=False`.

## Prompts

Cityscapes and ACDC use the same class order, names, and templates for a given model. ADE20K uses the 150 mmseg class names. The stored name `bed ` keeps its trailing space; the prompt noun is `bed`.

- SAN uses the mmseg ViLD template list and drops the learned background channel after the class softmax, which is `SideAdapterCLIPHead.predict_by_feat`.
- CAT-Seg (L) uses its config setting `PROMPT_ENSEMBLE_TYPE=single`: `A photo of a {} in the scene`.
- SED (L) uses its ImageNet template ensemble.
- MAFT+ uses its VILD template list.
- SAM 3 is prompted with the class name and no extra template.

## Attacks

Attacks are untargeted and white-box, for 20 iterations, on ADE20K only. They call `cospgd` `init_linf` / `init_l2` and `step_inf` / `step_l2`, and they use the SemSegBench SegPGD and CosPGD pixel weights. The loss is per-pixel cross-entropy on the dataset-class scores, with ignore index 255. SAN's official score is the fused map from `predict_by_feat` (class softmax, background dropped, times sigmoid masks). Pixel cross-entropy is applied to that tensor.

The attacked tensor is RGB float in `[0, 255]`. Linf uses epsilon 8 and step 2.55. L2 uses epsilon 64 on the full per-image perturbation and step 25.5. These are the benchmark configuration values `epsilon=8, alpha=0.01` and `epsilon=64, alpha=0.1` after the `[0, 255]` conversion in `EncoderDecoder.predict`. The scale is not inferred from `tensor.max()`.

## Corruptions

All 15 `imagecorruptions` corruptions are evaluated at severity 3 on ADE20K and Cityscapes. Corruption is applied to the benchmark-resolution image before model normalization. The image is converted RGB to BGR before `corrupt` and back to RGB afterward, because `SegDataPreProcessor` corrupts the OpenCV BGR tensor before `bgr_to_rgb`. Each image has a deterministic corruption seed so a resumed job reproduces it.

## ACDC aggregate

`mIoU_acdc` is the mIoU of the 406-image validation set. For each class, intersections and unions are summed over fog, night, rain, and snow, and the class IoU is that ratio. mIoU is the mean of those class IoUs, with the same absent-class handling as `IoUMetric`. It is not the mean of the four condition mIoUs.

## Metrics

mIoU, mAcc, and aAcc are percentages, rounded to two decimals as in `IoUMetric`. For ADE20K, `ReM_Linf` is the minimum of PGD, SegPGD, and CosPGD under Linf, and `ReM_L2` is the same minimum under L2. `GAM_3` is the minimum mIoU across the 15 severity-3 corruptions, computed separately for ADE20K and Cityscapes. These summaries are not extra inference runs.

## Environment

`scripts/vision_language_semseg/activate_task.sh` puts `semantic_segmentation/.vlseg_py` and this repository's `mmsegmentation` on `PYTHONPATH`. That directory holds stock mmengine 0.10.6. The shared OpenMMLab environment imports a RAW-image fork of mmengine, and that fork adds parameters that are not in these checkpoints, so it is not used for this extension.

The official SAN checkpoint loads with `strict=True` under that stock mmengine: 436,674,995 parameters, 150 ADE20K output channels, and frozen weights. A CPU forward and backward on a 128×128 tensor produced finite, nonzero image gradients. Full validation evaluations have not been run.

## Reproduction

From the repository root, after the model environment for the selected row exists:

```bash
export VLSEG_GROUP=clean
export SLURM_ARRAY_TASK_ID=0
bash scripts/vision_language_semseg/activate_task.sh
python scripts/vision_language_semseg/run_experiment.py
```

Slurm groups, from the repository root:

```bash
sbatch scripts/vision_language_semseg/clean.sbatch
sbatch scripts/vision_language_semseg/adversarial.sbatch
sbatch scripts/vision_language_semseg/corruptions.sbatch
sbatch scripts/vision_language_semseg/acdc.sbatch
```

The arrays are 0-11, 0-35, 0-179, and 0-29. Logs are `slurm/vision_language_semseg/<group>/%A_%a.out`. Jobs request one H100, 12 CPUs, and 100G on `gpu-vram-94gb`. These nodes set `RestrictedCoresPerGPU=12`, so 12 CPUs is the request that can pack four GPUs on a node. The account is `ml-staff` and the QoS is `walltime5d`.

A restarted task reloads `results/vision_language_semseg/runs/<experiment_id>/counts.npz`, keeps completed image ids, and recomputes a missing image from the same attack seed.
