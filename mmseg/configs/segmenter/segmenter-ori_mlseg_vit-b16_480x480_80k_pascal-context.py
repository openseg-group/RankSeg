_base_ = [
    # "./training_scheme.py",
    "../_base_/models/segmenter_vit-b16.py",
    "../_base_/datasets/pascal_context_meanstd0.5_test.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]

model = dict(
    backbone=dict(
        drop_path_rate=0.1,
        final_norm=True,
    ),
    neck=dict(
        type="UseIndexSingleOutNeck",
        index=-1,
    ),
    decode_head=dict(
        type="MaskTransformerMLSegHead",
        n_cls=60,
        topk_cls=25,
        downsample=8,
        img_loss_weight=5,
        loss_img=dict(type="AsymmetricLoss", gamma_neg=2, gamma_pos=0),
        n_layers=3,
    ),
    test_cfg=dict(mode="slide", crop_size=(480, 480), stride=(480, 480)),
)

optimizer = dict(
    _delete_=True,
    type="SGD",
    lr=0.001,
    weight_decay=0.000001,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            "pos_embed": dict(decay_mult=0.0),
            "cls_token": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "decode_head.blocks.0.": dict(decay_mult=100),
            "decode_head.ml_fc": dict(decay_mult=100),
        }
    ),
)

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup_iters=0,
    power=0.9,
    min_lr=1e-5,
    by_epoch=False,
)
# By default, models are trained on 8 GPUs with 1 images per GPU
data = dict(samples_per_gpu=1)
evaluation = dict(interval=8000, metric="mIoU", pre_eval=True)
