_base_ = [
    # "./training_scheme.py",
    "../_base_/models/segmenter_vit-b16.py",
    "../_base_/datasets/ade20k_640_meanstd0.5.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
]

model = dict(
    pretrained="pretrain/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz",
    backbone=dict(
        type="VisionTransformer",
        img_size=(640, 640),
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        out_indices=(5, 11, 17, 23),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cls_token=True,
        final_norm=True,
        norm_cfg=dict(type="LN", eps=1e-6),
        act_cfg=dict(type="GELU"),
        norm_eval=False,
        interpolate_mode="bicubic",
    ),
    neck=dict(
        type="UseIndexSingleOutNeck",
        index=-1,
    ),
    decode_head=dict(
        n_cls=150,
        d_encoder=1024,
        n_heads=16,
        d_model=1024,
        d_ff=4 * 1024,
    ),
    test_cfg=dict(mode="slide", crop_size=(640, 640), stride=(640, 640)),
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
