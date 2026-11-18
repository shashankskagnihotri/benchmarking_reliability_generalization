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
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        init_cfg=dict(
            checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
            type="Pretrained",
        ),
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type="SwinTransformer",
        window_size=7,
        with_cp=False,
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
        add_extra_convs="on_input",
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        num_outs=4,
        out_channels=256,
        start_level=0,
        type="FPN",
    ),
    roi_head=dict(
        bbox_head=[
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type="DeltaXYWHBBoxCoder",
                ),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                ),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type="ReLU"),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type="L1Loss"),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type="FocalLoss",
                    use_sigmoid=True,
                ),
                loss_iou=dict(loss_weight=2.0, type="GIoULoss"),
                num_classes=80,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type="DIIHead",
            ),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type="DeltaXYWHBBoxCoder",
                ),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                ),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type="ReLU"),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type="L1Loss"),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type="FocalLoss",
                    use_sigmoid=True,
                ),
                loss_iou=dict(loss_weight=2.0, type="GIoULoss"),
                num_classes=80,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type="DIIHead",
            ),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type="DeltaXYWHBBoxCoder",
                ),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                ),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type="ReLU"),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type="L1Loss"),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type="FocalLoss",
                    use_sigmoid=True,
                ),
                loss_iou=dict(loss_weight=2.0, type="GIoULoss"),
                num_classes=80,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type="DIIHead",
            ),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type="DeltaXYWHBBoxCoder",
                ),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                ),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type="ReLU"),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type="L1Loss"),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type="FocalLoss",
                    use_sigmoid=True,
                ),
                loss_iou=dict(loss_weight=2.0, type="GIoULoss"),
                num_classes=80,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type="DIIHead",
            ),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type="DeltaXYWHBBoxCoder",
                ),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                ),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type="ReLU"),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type="L1Loss"),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type="FocalLoss",
                    use_sigmoid=True,
                ),
                loss_iou=dict(loss_weight=2.0, type="GIoULoss"),
                num_classes=80,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type="DIIHead",
            ),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type="DeltaXYWHBBoxCoder",
                ),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                ),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type="ReLU"),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type="L1Loss"),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type="FocalLoss",
                    use_sigmoid=True,
                ),
                loss_iou=dict(loss_weight=2.0, type="GIoULoss"),
                num_classes=80,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type="DIIHead",
            ),
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=2, type="RoIAlign"),
            type="SingleRoIExtractor",
        ),
        num_stages=6,
        proposal_feature_channel=256,
        stage_loss_weights=[
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        type="SparseRoIHead",
    ),
    rpn_head=dict(
        num_proposals=300, proposal_feature_channel=256, type="EmbeddingRPNHead"
    ),
    test_cfg=dict(rcnn=dict(max_per_img=300), rpn=None),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type="FocalLossCost", weight=2.0),
                        dict(box_format="xyxy", type="BBoxL1Cost", weight=5.0),
                        dict(iou_mode="giou", type="IoUCost", weight=2.0),
                    ],
                    type="HungarianAssigner",
                ),
                pos_weight=1,
                sampler=dict(type="PseudoSampler"),
            ),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type="FocalLossCost", weight=2.0),
                        dict(box_format="xyxy", type="BBoxL1Cost", weight=5.0),
                        dict(iou_mode="giou", type="IoUCost", weight=2.0),
                    ],
                    type="HungarianAssigner",
                ),
                pos_weight=1,
                sampler=dict(type="PseudoSampler"),
            ),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type="FocalLossCost", weight=2.0),
                        dict(box_format="xyxy", type="BBoxL1Cost", weight=5.0),
                        dict(iou_mode="giou", type="IoUCost", weight=2.0),
                    ],
                    type="HungarianAssigner",
                ),
                pos_weight=1,
                sampler=dict(type="PseudoSampler"),
            ),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type="FocalLossCost", weight=2.0),
                        dict(box_format="xyxy", type="BBoxL1Cost", weight=5.0),
                        dict(iou_mode="giou", type="IoUCost", weight=2.0),
                    ],
                    type="HungarianAssigner",
                ),
                pos_weight=1,
                sampler=dict(type="PseudoSampler"),
            ),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type="FocalLossCost", weight=2.0),
                        dict(box_format="xyxy", type="BBoxL1Cost", weight=5.0),
                        dict(iou_mode="giou", type="IoUCost", weight=2.0),
                    ],
                    type="HungarianAssigner",
                ),
                pos_weight=1,
                sampler=dict(type="PseudoSampler"),
            ),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type="FocalLossCost", weight=2.0),
                        dict(box_format="xyxy", type="BBoxL1Cost", weight=5.0),
                        dict(iou_mode="giou", type="IoUCost", weight=2.0),
                    ],
                    type="HungarianAssigner",
                ),
                pos_weight=1,
                sampler=dict(type="PseudoSampler"),
            ),
        ],
        rpn=None,
    ),
    type="SparseRCNN",
)
num_proposals = 300
num_stages = 6
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=0.0001,
        type="AdamW",
        weight_decay=0.05,
    ),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
        )
    ),
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
            dict(prob=0.5, type="RandomFlip"),
            dict(
                transforms=[
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type="RandomChoiceResize",
                        ),
                    ],
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    400,
                                    1333,
                                ),
                                (
                                    500,
                                    1333,
                                ),
                                (
                                    600,
                                    1333,
                                ),
                            ],
                            type="RandomChoiceResize",
                        ),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(
                                384,
                                600,
                            ),
                            crop_type="absolute_range",
                            type="RandomCrop",
                        ),
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type="RandomChoiceResize",
                        ),
                    ],
                ],
                type="RandomChoice",
            ),
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
    dict(prob=0.5, type="RandomFlip"),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type="RandomChoiceResize",
                ),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            1333,
                        ),
                        (
                            500,
                            1333,
                        ),
                        (
                            600,
                            1333,
                        ),
                    ],
                    type="RandomChoiceResize",
                ),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type="absolute_range",
                    type="RandomCrop",
                ),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type="RandomChoiceResize",
                ),
            ],
        ],
        type="RandomChoice",
    ),
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
            config=dict(config_name="sparse_rcnn_swin-s_coco"), project="Training"
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
                config=dict(config_name="sparse_rcnn_swin-s_coco"), project="Training"
            ),
            type="WandbVisBackend",
        ),
    ],
)
work_dir = "./slurm/train_work_dir/sparse_rcnn_swin-s_coco"
