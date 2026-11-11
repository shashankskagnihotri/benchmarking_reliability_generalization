# Model provenance

Full segmentation table: [`results/model_provenance.csv`](../results/model_provenance.csv).

One row is one checkpoint. `model_id` is `architecture_backbone_dataset_crop`. The same architecture and backbone on one dataset can be two checkpoints when the crop differs (Cityscapes SegFormer at 512×1024 and 1024×1024; ADE20K Mask2Former Swin-B at 512×512 and 640×640).

`origin` is decided from files in this repository:

- `publicly released` — the clean mIoU matches a row in the MMSegmentation config README, or the only config for that checkpoint is that published file. The checkpoint URL is the README download link.
- `trained by authors` — an uncommented line in `semantic_segmentation/scripts/train_*.sh` launches that config, and the clean mIoU does not match a published checkpoint.
- `training config present; launch not confirmed` — the config is in the tree, but no uncommented script launches it.
- `not confirmed` — the closest file disagrees with the dataset (InternImage Cityscapes and VOC configs still set `num_classes=150`).

Epoch counts are not in these configs. Schedules are iteration-based. Augmentation is the `train_pipeline` reached through `_base_`.

Checkpoints are not in git. `get_checkpoint_path` reads `semantic_segmentation/checkpoints/<architecture>/*.pth`. ResNet names are matched as `r50`, `r101`, or `r18`. Other backbones are matched as written. The filename must also contain the dataset. `load_model` opens the config path in [`results/model_provenance.csv`](../results/model_provenance.csv) for that architecture, backbone, dataset, and crop. If the local checkpoint is missing, it falls back to the MMSegmentation config name, which downloads a published checkpoint when one is registered.

## Models trained for this study

Weights for these runs are not in the repository. Backbone initialization URLs are in the config files and in the CSV (`pretraining`).

| Architecture | Backbone | Dataset | Crop | Iterations | Optimizer | Training config |
| --- | --- | --- | --- | --- | --- | --- |
| SegFormer | MiT-B0 | Cityscapes | 512×1024 | 160k | AdamW, lr 6e-5 | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-512x1024.py) |
| SegFormer | MiT-B2 | Cityscapes | 512×1024 | 160k | AdamW, lr 6e-5, inherited from the B0 file | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-512x1024.py) |
| SegFormer | MiT-B3 | Cityscapes | 512×1024 | 160k | AdamW, lr 6e-5, inherited from the B0 file | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/segformer/segformer_mit-b3_8xb1-160k_cityscapes-512x1024.py) |
| SegFormer | MiT-B4 | Cityscapes | 512×1024 | 160k | AdamW, lr 6e-5, inherited from the B0 file | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/segformer/segformer_mit-b4_8xb1-160k_cityscapes-512x1024.py) |
| SegFormer | MiT-B5 | Cityscapes | 512×1024 | 160k | AdamW, lr 6e-5, inherited from the B0 file | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-512x1024.py) |
| Mask2Former | Swin-B, ImageNet-1k | Cityscapes | 512×1024 | 90k | AdamW, lr 1e-4 | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/mmsegmentation/configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-90k_cityscapes-512x1024.py) |
| Mask2Former | Swin-B, ImageNet-1k | ADE20K | 512×512 | 160k | AdamW, lr 1e-4 | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-512x512.py) |
| SegFormer | MiT-B0 | PASCAL VOC | 512×512 | 160k | AdamW, lr 6e-5 | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/segformer/segformer_mit-b0_8xb2-160k_voc12aug-512x512.py) |
| SegFormer | MiT-B1 | PASCAL VOC | 512×512 | 160k | AdamW, lr 6e-5, inherited from the B0 file | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/segformer/segformer_mit-b1_8xb2-160k_voc12aug-512x512.py) |
| SegFormer | MiT-B2 | PASCAL VOC | 512×512 | 160k | AdamW, lr 6e-5, inherited from the B0 file | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/segformer/segformer_mit-b2_8xb2-160k_voc12aug-512x512.py) |
| SegFormer | MiT-B3 | PASCAL VOC | 512×512 | 160k | AdamW, lr 6e-5, inherited from the B0 file | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/segformer/segformer_mit-b3_8xb2-160k_voc12aug-512x512.py) |
| SegFormer | MiT-B4 | PASCAL VOC | 512×512 | 160k | AdamW, lr 6e-5, inherited from the B0 file | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/segformer/segformer_mit-b4_8xb2-160k_voc12aug-512x512.py) |
| SegFormer | MiT-B5 | PASCAL VOC | 512×512 | 160k | AdamW, lr 6e-5, inherited from the B0 file | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/segformer/segformer_mit-b5_8xb2-160k_voc12aug-512x512.py) |
| Mask2Former | Swin-S | PASCAL VOC | 512×512 | 160k | AdamW, lr 1e-4 | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/mask2former/mask2former_swin-s_8xb2-160k_voc12aug-512x512.py) |
| Mask2Former | Swin-B, ImageNet-1k | PASCAL VOC | 512×512 | 160k | AdamW, lr 1e-4 | [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_voc12aug-512x512.py) |

Cityscapes SegFormer B2–B5 and VOC SegFormer B1–B5 set the backbone checkpoint and width, and inherit the optimizer from the matching B0 config. Mask2Former Swin-B Cityscapes is the ImageNet-1k config. Its clean mIoU (80.03) is not the published ImageNet-22k checkpoint (83.52).

`scripts/train_voc12aug_segformer_mit-b1-5_512x512.sh` points at 40k configs (`segformer_mit-b*_8xb2-40k_voc12aug-512x512.py`). Those files are not in the repository. The 160k configs above are the ones that exist and that `scripts/train_voc12aug_segformer_160k_mit-b0-5_512x512.sh` launches.

### Config present, launch not confirmed

These have a training config and no uncommented script that starts it. They are not listed as author-trained above.

- SegFormer MiT-B1, Cityscapes, 512×1024: [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/segformer/segformer_mit-b1_8xb1-160k_cityscapes-512x1024.py). `train_cityscapes_segformer_mit-b1-5_512x1024.sh` starts at MiT-B2.
- Mask2Former Swin-T, PASCAL VOC: [config](https://github.com/shashankskagnihotri/benchmarking_reliability_generalization/blob/main/semantic_segmentation/configs/mask2former/mask2former_swin-t_8xb2-160k_voc12aug-512x512.py). The launch line in `scripts/train_pascal_mask2former.sh` is commented out.

## Object detection

Appendix K says a few MS-COCO detectors were trained for the study and the rest are public checkpoints. The appendix does not name which ones were trained, and this repository has no detector training configs, checkpoint URLs, or result files.

Appendix M lists the evaluated detectors. That list is transcribed, without invented training fields, in [`results/detection_models_manuscript.csv`](../results/detection_models_manuscript.csv) (62 architecture–backbone pairs). The paper text says 61 detectors. The per-evaluation logs that would reconcile 61 with 4,087 are not in this repository.

DetecBench loads whatever `.py` and `.pth` are in the folder passed as `model_folder` (see `object_detection/src/detecbench/detecbench.py`). There is no `download_models.py`.

A single training recipe for every detector was not run. That comparison is outside this release.
