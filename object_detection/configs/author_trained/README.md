# Author-trained MS-COCO detectors

Training configs for the ten MS-COCO detectors trained for this study, from `benchmarking_robustness` branch `object_detection_all`, `object_detection/horeka_test_submission/train_fully/`.

Each file is a full MMDetection config. COCO is at `data/coco/`:

- `annotations/instances_train2017.json`
- `annotations/instances_val2017.json`
- `train2017/`
- `val2017/`

PAA, Sparse R-CNN, and TOOD write to `./slurm/train_work_dir/<config name>`.

`model.backbone.init_cfg.checkpoint` is the ImageNet backbone initialization. Training runs for 36 epochs, with early stopping on `coco/bbox_mAP`.
