# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
from mmcv.cnn import (
    build_norm_layer,
    constant_init,
    kaiming_init,
    normal_init,
    trunc_normal_init,
)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.ops import resize
from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbed
from ...ops.ssn_layers import SSNIterations, get_aspect_ratio, reshape_as_aspect_ratio
from ...ops.light_conv_encoder import LightConv_Encoder
from ...ops.unpooling_layers import NaiveUnpooling, MHAUnpooling

from .vit import VisionTransformer


@BACKBONES.register_module()
class VisionTransformerSSN(VisionTransformer):
    """
    Extensible ViT for pluggable stems or SSN layers
    """

    def __init__(self, *, stem="PatchEmbed", ssn_cfg=None, unpool_cfg=None, **kwds):
        super().__init__(**kwds)

        # self.aspect_ratio = get_aspect_ratio(*self.img_size)
        self.build_stem(stem, kwds)
        self.build_ssn(ssn_cfg)
        self.build_unpooling(unpool_cfg, kwds)

    def build_stem(self, stem_type: str, kwds: dict):
        if stem_type == "PatchEmbed":
            # already built in ancestor's __init__, noop here
            pass
        elif stem_type == "Conv7GF16x":
            self.patch_embed = LightConv_Encoder(
                in_channels=3,
                out_channels=kwds["embed_dims"],
                hidden_channels=[64, 128, 256, 512],
                img_size=self.img_size,
            )
        else:
            raise KeyError(f"unknown stem type: {stem_type!r}")

    def build_ssn(self, cfg: dict):
        assert cfg is not None
        niters = cfg.get("niters", 5)
        stages = cfg.get("stages", [])

        ssns = []
        accum = 0
        for stage in stages:
            num_blocks = stage["num_blocks"]
            num_patches = stage["num_patches"]
            accum += num_blocks
            ssns.append((str(accum), SSNIterations(num_patches, niters)))

        self.ssn_blocks = nn.ModuleDict(OrderedDict(ssns))
        self.ssn_after_output = cfg.get("after_output", True)

    def build_unpooling(self, cfg: dict, kwds: dict):
        assert cfg is not None
        type_ = cfg.get("type")

        if type_ == "naive":
            module = NaiveUnpooling()
        elif type_ == "MHA":
            module = MHAUnpooling(embed_dim=kwds["embed_dims"], cfg=cfg)
        else:
            raise KeyError(f"unknown unpooling type: {type_!r}")

        self.unpooling = module

    def forward_ssn_block(self, x, block_idx: int, unpooler):
        ssn_key = str(block_idx + 1)
        if ssn_key not in self.ssn_blocks:
            return x
        cls_tokens = x[:, 0:1]
        unpooler.update_state(feat_before_pooling=x[:, 1:])
        x = reshape_as_aspect_ratio(x[:, 1:], unpooler.aspect_ratio)
        x, hard_labels = self.ssn_blocks[ssn_key](x)
        unpooler.update_state(hard_labels=hard_labels)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

    def forward(self, inputs):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs), (
            self.patch_embed.DH,
            self.patch_embed.DW,
        )
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        aspect_ratio = get_aspect_ratio(*hw_shape)
        outs = []
        unpooler = self.unpooling.derive_unpooler()
        unpooler.aspect_ratio = aspect_ratio
        unpooler.hw_shape = hw_shape
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
            if not self.ssn_after_output:
                x = self.forward_ssn_block(x, i, unpooler)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                out, reshaped = unpooler.call(out)
                if not reshaped:
                    out = reshape_as_aspect_ratio(out, aspect_ratio)
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)
            if self.ssn_after_output:
                x = self.forward_ssn_block(x, i, unpooler)

        return tuple(outs)
