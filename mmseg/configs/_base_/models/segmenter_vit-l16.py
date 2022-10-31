# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
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
    decode_head=dict(
        type="MaskTransformerHead",
        n_cls=150,
        d_encoder=1024,
        patch_size=16,
        n_heads=16,
        n_layers=2,
        d_model=1024,
        d_ff=4 * 1024,
        drop_path_rate=0.0,
        dropout=0.1,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),

    test_cfg=dict(mode="slide", crop_size=(640, 640), stride=(640, 640)),
    train_cfg={}
)
  # yapf: disable

