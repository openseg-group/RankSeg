import torch
import torch.nn as nn
import math

from .multi_head_attention import MultiheadAttention
from .ssn_layers import reshape_as_aspect_ratio

from mmcv.cnn import ConvModule, xavier_init, DepthwiseSeparableConvModule
from mmcv.cnn import (
    build_norm_layer,
    constant_init,
    kaiming_init,
    normal_init,
    trunc_normal_init,
)
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm


def naive_unpool(f_regions, region_indices):
    _, _, C = f_regions.shape
    N, L = region_indices.shape
    index = region_indices.view(N, L, 1).expand(N, L, C)
    result = f_regions.gather(1, index)
    return result


class State:
    def __init__(self, unpooling):
        self.unpooling = unpooling
        self.__updated = False

    @property
    def updated(self):
        return self.__updated

    def get(self, name, default=None):
        return getattr(self, name, default)

    def update_state(self, **states: dict):
        self.__updated = True
        for k, v in states.items():
            setattr(self, k, v)

    def call(self, input: torch.Tensor):
        return self.unpooling(input, self)


class UnpoolingBase(nn.Module):
    def forward(self, x, state: State):
        if not state.updated:
            return x, False
        return self._forward(x, state)

    def derive_unpooler(self):
        return State(self)


class NaiveUnpooling(UnpoolingBase):
    def _forward(self, x, state: State):
        return naive_unpool(x, state.hard_labels), False


class MHAUnpooling(UnpoolingBase):
    def __init__(self, *, embed_dim=768, cfg=None):
        super().__init__()

        assert isinstance(cfg, dict)
        num_blocks = cfg.get("num")
        assert isinstance(num_blocks, int)

        init_kwds = cfg.get("init")
        assert isinstance(init_kwds, dict)
        init_kwds.update(embed_dim=embed_dim)

        overrides = cfg.get("overrides", [{}] * num_blocks)
        assert len(overrides) == num_blocks

        self.refine = cfg.get("refine", False)

        fuse = cfg.get("fuse", {})
        self.fuse_type = fuse_type = fuse.get("type", "none")
        assert fuse_type in {"none", "cat", "add"}

        self.fuse_source = fuse_source = fuse.get("source", "MHA")
        assert fuse_source in {"MHA", "orig"}

        if fuse_type != "none":
            fuse_trans_cfg = fuse.get("transform", {})
            fuse_trans_builder = fuse_trans_cfg.pop("builder")
            assert fuse_trans_builder in {"conv", "sep_conv"}
            conv_builder = {
                "conv": ConvModule,
                "sep_conv": DepthwiseSeparableConvModule,
            }[fuse_trans_builder]

            fuse_trans_cfg["in_channels"] = (
                embed_dim * 2 if fuse_type == "cat" else embed_dim
            )
            fuse_trans_cfg["out_channels"] = embed_dim
            fuse_trans_cfg.setdefault("act_cfg", None)
            self.fuse_trans = conv_builder(**fuse_trans_cfg)

        if fuse_source == "MHA":
            blocks = []
            for idx in range(num_blocks):
                kwds = init_kwds.copy()
                kwds.update(overrides[idx])
                blocks.append(MultiheadAttention(**kwds))

            self.blocks = nn.ModuleList(blocks)

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m.weight, std=0.02)
                if m.bias is not None:
                    if "ffn" in n:
                        normal_init(m.bias, std=1e-6)
                    else:
                        constant_init(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                kaiming_init(m.weight, mode="fan_in")
                if m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)

    def _forward(self, x, state: State):
        if self.fuse_type != "none":
            naive_ret = naive_unpool(x, state.hard_labels)

        query = state.feat_before_pooling

        if self.fuse_source == "orig":
            ret = query

        elif self.fuse_source == "MHA":
            block_idx = state.get("block_idx", 0)
            block = self.blocks[block_idx]
            
            query = query.permute(1, 0, 2)
            x = x.permute(1, 0, 2)

            ret = block(query, x, x, q_shape=state.hw_shape)

            state.block_idx = block_idx + 1

            ret = ret.permute(1, 0, 2)
        else:
            assert False

        if self.fuse_type != "none":
            if self.fuse_type == "cat":
                ret = torch.cat([ret, naive_ret], dim=2)
            elif self.fuse_type == "add":
                ret = ret + naive_ret
            else:
                assert False

            ret = reshape_as_aspect_ratio(ret, state.aspect_ratio)
            ret = self.fuse_trans(ret)
            reshaped = True
        else:
            reshaped = False

        if self.refine:
            tmp_ret = ret
            if tmp_ret.ndim == 4:
                N, C, H, W = tmp_ret.size()
                tmp_ret = tmp_ret.view(N, C, -1).permute(0, 2, 1)
            state.feat_before_pooling = tmp_ret

        return ret, reshaped