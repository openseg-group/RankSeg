import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import (
    ConvModule,
    DepthwiseSeparableConvModule,
    xavier_init,
    constant_init,
)

from ..builder import NECKS


@NECKS.register_module()
class UseIndexSingleOutNeck(nn.Module):
    def __init__(self, index=-1):
        super().__init__()
        self.index = index

    def forward(self, outs):
        return outs[-1]


@NECKS.register_module()
class FusionSingleOutNeck(nn.Module):
    BUILDER_OPTIONS = dict(
        ConvModule=ConvModule,
        DepthwiseSeparableConvModule=DepthwiseSeparableConvModule,
    )

    def __init__(self, builder="ConvModule", fusion="cat", **kwds):
        super().__init__()
        assert fusion in {"cat", "add"}
        self.fusion = fusion
        assert builder in self.BUILDER_OPTIONS
        kwds.setdefault("norm_cfg", None)
        kwds.setdefault("act_cfg", None)
        self.conv = self.BUILDER_OPTIONS[builder](**kwds)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")
            elif isinstance(m, (_BatchNorm,)):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)

    def forward(self, outs):
        if self.fusion == "cat":
            outs = torch.cat(outs, dim=1)
        elif self.fusion == "add":
            outs = sum(outs)
        return self.conv(outs)
