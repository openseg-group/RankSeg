_base_ = [
    # "./training_scheme.py",
    "../_base_/models/segmenter_vit-b16.py",
    "../_base_/datasets/coco-stuff10k.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
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
        type="MaskTransformerMLSegDecoderHead",
        n_cls=171,
        topk_cls=50,
        downsample=16,
        img_loss_weight=10,
        img_cls_head=dict(
            type="MLSegDecoderHead",
            input_dim=768,
            hidden_dim=768,
            size=16,
            num_class=171,
            share_embedding=True,
        ),
        loss_img=dict(type="AsymmetricLoss", gamma_neg=0, gamma_pos=0),       
    ),
    test_cfg=dict(mode="slide", crop_size=(512, 512), stride=(512, 512)),
)

optimizer = dict(
    _delete_=True,
    type="SGD",
    lr=0.001,
    weight_decay=0.0,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            "pos_embed": dict(decay_mult=0.0),
            "cls_token": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
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
