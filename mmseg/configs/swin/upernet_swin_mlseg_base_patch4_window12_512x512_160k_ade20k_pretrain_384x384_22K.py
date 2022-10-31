_base_ = [
    '../_base_/models/upernet_swin_mlseg.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    pretrained='pretrain/swin_base_patch4_window12_384_22k.pth',
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
        ),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=150, mlseg=dict(head=dict(num_classes=150))),
    auxiliary_head=dict(in_channels=512, num_classes=150))


optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            "decode_head.img_cls_head.block.": dict(lr_mult=0.1),
            "decode_head.img_cls_head.fc": dict(lr_mult=0.1)
        }))