batch_size = 10
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'
data_root = '/content/data/cityscapes/'
dataset_type = 'CityscapesDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
image_size = (
    384,
    384,
)
iters = 10000
launcher = 'none'
learning_rate = 0.001
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='ReLU'),
        align_corners=False,
        channels=32,
        in_channels=3,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_branch_blocks=3,
        num_stem_blocks=2,
        ppm_channels=96,
        type='PIDNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            384,
            384,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        act_cfg=dict(inplace=True, type='ReLU'),
        align_corners=True,
        channels=128,
        in_channels=128,
        loss_decode=[
            dict(loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
            dict(
                loss_weight=1.0,
                min_kept=131072,
                thres=0.9,
                type='OhemCrossEntropy'),
            dict(loss_weight=20.0, type='BoundaryLoss'),
            dict(
                loss_weight=1.0,
                min_kept=131072,
                thres=0.9,
                type='OhemCrossEntropy'),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        type='PIDHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=10000,
        eta_min=0,
        power=0.9,
        type='PolyLR'),
]
randomness = dict(seed=304)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            data_root='/content/data/cityscapes/',
            img_path='leftImg8bit/val',
            seg_map_path='gtFine/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                384,
                384,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
train_cfg = dict(max_iters=10000, type='IterBasedTrainLoop', val_interval=500)
train_dataloader = dict(
    batch_size=10,
    dataset=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root='/content/data/cityscapes/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    768,
                    384,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    384,
                    384,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(edge_width=4, type='GenerateEdge'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
            dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
            dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
            dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
            dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
            dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            dict(direction='horizontal', prob=0.0, type='RandomFlip'),
            dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root='/content/data/cityscapes/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                384,
                384,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CityscapesDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = '/content/work_dir'
