import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import trunc_normal_
from mmcv.cnn import ConvModule

from ..builder import HEADS, build_loss
from .decode_head import BaseDecodeHead


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


@HEADS.register_module()
class MaskTransformerEncHead(BaseDecodeHead):

    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
        num_codes=32,
        loss_se_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.2),
        layer_norm="orig",
        **kwargs,
    ):
        # in_channels & channels are dummy arguments to satisfy signature of
        # parent's __init__
        super().__init__(
            in_channels=d_encoder,
            channels=1,
            num_classes=n_cls,
            **kwargs,
        )
        del self.conv_seg
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model**-0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, dpr[i])
            for i in range(n_layers)
        ])

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale *
                                       torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale *
                                         torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        if layer_norm == "orig":
            self.mask_norm = nn.LayerNorm(n_cls)
        elif layer_norm == "scale":
            self.mask_norm = LayerScale(n_cls)
        self.layer_norm = layer_norm

        self.loss_se_decode = build_loss(loss_se_decode)
        self.se_layer = nn.Linear(self.d_model, self.num_classes)
        from .enc_head import EncModule
        self.enc_module = EncModule(
            self.d_model,
            num_codes=num_codes,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            self.d_encoder,
            self.d_model,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def init_weights(self):
        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, x):
        x = self._transform_inputs(x)
        x = self.bottleneck(x)
        encode_feat, x = self.enc_module(x)

        encode_feat = self.se_layer(encode_feat)

        B, C, H, W = x.size()
        # GS = H // self.patch_size
        x = x.view(B, C, -1).permute(0, 2, 1)

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, :-self.n_cls], x[:, -self.n_cls:]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        if self.layer_norm:
            masks = self.mask_norm(masks)
        B, HW, N = masks.size()

        masks = masks.view(B, H, W, N).permute(0, 3, 1, 2)
        # masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        if self.training:
            return masks, encode_feat

        else:
            return masks

    @staticmethod
    def _convert_to_onehot_labels(seg_label, num_classes):
        """Convert segmentation label to onehot.

        Args:
            seg_label (Tensor): Segmentation label of shape (N, H, W).
            num_classes (int): Number of classes.

        Returns:
            Tensor: Onehot labels of shape (N, num_classes).
        """

        batch_size = seg_label.size(0)
        onehot_labels = seg_label.new_zeros((batch_size, num_classes))
        for i in range(batch_size):
            hist = seg_label[i].float().histc(
                bins=num_classes, min=0, max=num_classes - 1)
            onehot_labels[i] = hist > 0
        return onehot_labels

    def losses(self, seg_logit, seg_label):
        """Compute segmentation and semantic encoding loss."""
        seg_logit, se_seg_logit = seg_logit
        loss = dict()
        loss.update(
            super(MaskTransformerEncHead, self).losses(seg_logit, seg_label))
        se_loss = self.loss_se_decode(
            se_seg_logit,
            self._convert_to_onehot_labels(seg_label, self.num_classes))
        loss['loss_se'] = se_loss
        return loss


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim**-0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.heads,
                                C // self.heads).permute(2, 0, 3, 1, 4))
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):

    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderLinear(nn.Module):

    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        B, HW, C = x.size()
        x = x.view(B, GS, HW // GS, C).permute(0, 3, 1, 2)
        # x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x


class LayerScale(nn.Module):
    '''Reproduced LayerNorm
    https://github.com/pytorch/pytorch/blob/e2eb97dd7682d2810071ce78b76543acc1584a9c/torch/onnx/symbolic_opset9.py#L1311
    '''

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return self.gamma * x + self.beta