find_unused_parameters = True
auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
custom_hooks = [
    dict(monitor="coco/bbox_mAP", patience=15, type="EarlyStoppingHook"),
]
data_root = "data/coco/"
dataset_type = "CocoDataset"
default_hooks = dict(
    checkpoint=dict(interval=1, type="CheckpointHook"),
    logger=dict(interval=50, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(type="DetVisualizationHook"),
)
default_scope = "mmdet"
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)
launcher = "pytorch"
load_from = None
log_level = "INFO"
log_processor = dict(by_epoch=True, type="LogProcessor", window_size=50)
max_epochs = 36
model = dict(
    backbone=dict(
        arch="small",
        drop_path_rate=0.6,
        gap_before_final_norm=False,
        init_cfg=dict(
            checkpoint="https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth",
            prefix="backbone.",
            type="Pretrained",
        ),
        layer_scale_init_value=1.0,
        out_indices=[
            0,
            1,
            2,
            3,
        ],
        type="mmpretrain.ConvNeXt",
    ),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=8,
            ratios=[
                1.0,
            ],
            scales_per_octave=1,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type="AnchorGenerator",
        ),
        anchor_type="anchor_free",
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                0.1,
                0.1,
                0.2,
                0.2,
            ],
            type="DeltaXYWHBBoxCoder",
        ),
        feat_channels=256,
        in_channels=256,
        initial_loss_cls=dict(
            activated=True,
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type="FocalLoss",
            use_sigmoid=True,
        ),
        loss_bbox=dict(loss_weight=2.0, type="GIoULoss"),
        loss_cls=dict(
            activated=True,
            beta=2.0,
            loss_weight=1.0,
            type="QualityFocalLoss",
            use_sigmoid=True,
        ),
        num_classes=80,
        stacked_convs=6,
        type="TOODHead",
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type="DetDataPreprocessor",
    ),
    neck=dict(
        add_extra_convs="on_output",
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type="FPN",
    ),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type="nms"),
        nms_pre=1000,
        score_thr=0.05,
    ),
    train_cfg=dict(
        allowed_border=-1,
        alpha=1,
        assigner=dict(topk=13, type="TaskAlignedAssigner"),
        beta=6,
        debug=False,
        initial_assigner=dict(topk=9, type="ATSSAssigner"),
        initial_epoch=4,
        pos_weight=-1,
    ),
    type="TOOD",
)
optim_wrapper = dict(
    constructor="LearningRateDecayOptimizerConstructor",
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=0.0002,
        type="AdamW",
        weight_decay=0.05,
    ),
    paramwise_cfg=dict(decay_rate=0.7, decay_type="layer_wise", num_layers=12),
    type="OptimWrapper",
)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=0.001, type="LinearLR"),
    dict(
        begin=0,
        by_epoch=True,
        end=36,
        gamma=0.1,
        milestones=[
            27,
            33,
        ],
        type="MultiStepLR",
    ),
]
resume = True
test_cfg = dict(type="TestLoop")
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file="annotations/instances_val2017.json",
        backend_args=None,
        data_prefix=dict(img="val2017/"),
        data_root="data/coco/",
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type="Resize",
            ),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
test_evaluator = dict(
    ann_file="data/coco/annotations/instances_val2017.json",
    backend_args=None,
    format_only=False,
    metric="bbox",
    type="CocoMetric",
)
test_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(
        keep_ratio=True,
        scale=(
            1333,
            800,
        ),
        type="Resize",
    ),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
        type="PackDetInputs",
    ),
]
train_cfg = dict(max_epochs=36, type="EpochBasedTrainLoop", val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    batch_size=2,
    dataset=dict(
        ann_file="annotations/instances_train2017.json",
        backend_args=None,
        data_prefix=dict(img="train2017/"),
        data_root="data/coco/",
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                keep_ratio=True,
                scale=[
                    (
                        1333,
                        480,
                    ),
                    (
                        1333,
                        800,
                    ),
                ],
                type="RandomResize",
            ),
            dict(prob=0.5, type="RandomFlip"),
            dict(type="PackDetInputs"),
        ],
        type="CocoDataset",
    ),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)
train_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        keep_ratio=True,
        scale=[
            (
                1333,
                480,
            ),
            (
                1333,
                800,
            ),
        ],
        type="RandomResize",
    ),
    dict(prob=0.5, type="RandomFlip"),
    dict(type="PackDetInputs"),
]
val_cfg = dict(type="ValLoop")
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file="annotations/instances_val2017.json",
        backend_args=None,
        data_prefix=dict(img="val2017/"),
        data_root="data/coco/",
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type="Resize",
            ),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
val_evaluator = dict(
    ann_file="data/coco/annotations/instances_val2017.json",
    backend_args=None,
    format_only=False,
    metric="bbox",
    type="CocoMetric",
)
vis_backends = [
    dict(
        init_kwargs=dict(
            config=dict(config_name="tood_convnext-s_coco"), project="Training"
        ),
        type="WandbVisBackend",
    ),
]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[
        dict(
            init_kwargs=dict(
                config=dict(config_name="tood_convnext-s_coco"), project="Training"
            ),
            type="WandbVisBackend",
        ),
    ],
)
work_dir = "./slurm/train_work_dir/tood_convnext-s_coco"
