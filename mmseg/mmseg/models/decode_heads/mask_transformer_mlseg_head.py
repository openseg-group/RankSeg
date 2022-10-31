import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import trunc_normal_
from timm.models.layers import DropPath
from mmseg.ops import resize

from ..builder import HEADS, build_loss
from .decode_head import BaseDecodeHead
from .mlseg_decoder_head import GroupWiseLinear
from ..losses.accuracy import (
    accuracy,
    img_recall,
    pixel_recall,
    img_accuracy,
    get_average_precision,
)


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


@HEADS.register_module()
class MaskTransformerMLSegHead(BaseDecodeHead):

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
            topk_cls,
            img_loss_weight,
            downsample=None,
            cls_emb_from_backbone=False,
            loss_img=dict(type="BCELoss"),
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
        self.downsample = downsample
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.loss_img = build_loss(loss_img)
        self.topk_cls = topk_cls
        self.scale = d_model**-0.5
        self.img_loss_weight = img_loss_weight

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, dpr[i])
            for i in range(n_layers)
        ])

        self.cls_emb_from_backbone = cls_emb_from_backbone
        if not cls_emb_from_backbone:
            self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale *
                                       torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale *
                                         torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm2 = nn.LayerNorm(d_model)

        if downsample:
            self.pooling = nn.AdaptiveAvgPool2d(downsample)

        self.mask_norm = nn.LayerNorm(topk_cls)

        self.ml_fc = GroupWiseLinear(n_cls, d_model)

    def init_weights(self):
        self.apply(init_weights)
        if not self.cls_emb_from_backbone:
            trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, x):
        if self.cls_emb_from_backbone:
            x, cls_emb = x
        else:
            cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = self._transform_inputs(x)
        B, C, H, W = x.size()

        # Downsample Input for Multi-Label Classification
        if self.downsample:
            downsample_x = self.pooling(x).view(B, C, -1).permute(0, 2, 1)
            downsample_x = self.proj_dec(downsample_x)

        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.proj_dec(x)

        if self.downsample:
            patch_and_cls_feat = torch.cat((downsample_x, cls_emb), 1)
            patch_and_cls_feat = self.blocks[0](patch_and_cls_feat)
            cls_ml_feat = patch_and_cls_feat[:, -self.n_cls:]
            pixel_feat = x
        else:
            patch_and_cls_feat = torch.cat((x, cls_emb), 1)
            patch_and_cls_feat = self.blocks[0](patch_and_cls_feat)
            cls_ml_feat = patch_and_cls_feat[:, -self.n_cls:]
            pixel_feat = patch_and_cls_feat[:, :-self.n_cls]

        cls_ml_feat = self.decoder_norm2(cls_ml_feat)
        img_pred = self.ml_fc(cls_ml_feat)

        topk = self.topk_cls
        topk_index = (
            torch.argsort(img_pred, dim=1,
                          descending=True)[:, :topk].squeeze(-1).squeeze(-1))

        # Select topK classes feature for every image
        cls_topk_feat = cls_ml_feat.view(self.n_cls,
                                         -1)[topk_index.view(-1)].view(
                                             B, topk, -1)
        patch_and_cls_feat = torch.cat((pixel_feat, cls_topk_feat), dim=1)

        # (nlayer - 1) layers of self attention
        for blk in self.blocks[1:]:
            patch_and_cls_feat = blk(patch_and_cls_feat)
        patch_and_cls_feat = self.decoder_norm(patch_and_cls_feat)

        patches, cls_seg_feat = (
            patch_and_cls_feat[:, :-topk],
            patch_and_cls_feat[:, -topk:],
        )

        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)

        B, HW, N = masks.size()
        masks = masks.view(B, H, W, N).permute(0, 3, 1, 2)

        # return pixel prediction in topK classes during training
        # gt_labels are rearranged in self.losses() with topk_index
        if self.training:
            return img_pred, topk_index, masks

        # return pixel prediction in all classes during inference
        # set the logits of classes not in topK classes to -100
        else:
            topk_index_tmp = topk_index.unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, H, W)
            full_masks = masks.new_ones((B, self.n_cls, H, W)) * -100
            full_masks.scatter_(1, topk_index_tmp, masks)
            return full_masks

    @staticmethod
    def _get_batch_hist_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch, H, W = target.shape
        tvect = target.new_zeros((batch, nclass), dtype=torch.int64)
        for i in range(batch):
            hist = torch.histc(
                target[i].data.float(), bins=nclass, min=0, max=nclass - 1)
            tvect[i] = hist
        return tvect

    def losses(self, outputs, seg_label):
        """Compute segmentation loss."""
        # generate histogram of gt_labels for each image
        hist = self._get_batch_hist_vector(
            seg_label.squeeze(1), self.num_classes)
        img_pred, topk_index, pixel_pred = outputs

        loss = dict()
        # Image Multi-label Loss
        loss["loss_img"] = (
            self.loss_img(
                img_pred.squeeze(-1).squeeze(-1),
                (hist > 0).float()) * self.img_loss_weight)

        # Seg Loss
        seg_logit = resize(
            input=pixel_pred,
            size=seg_label.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        # rearrange gt_labels to topK index, pixel not predicted in topK classes will be set to ignore
        topk_vector = (
            topk_index.unsqueeze(-1).unsqueeze(-1) == seg_label.repeat(
                1, self.topk_cls, 1, 1)).long()
        topk_max_prob, topk_label = torch.max(topk_vector, dim=1)
        topk_label[topk_max_prob == 0] = self.ignore_index
        topk_label[seg_label.squeeze(1) ==
                   self.ignore_index] = self.ignore_index

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, topk_label)
        else:
            seg_weight = None

        topk_label = topk_label.squeeze(1)
        loss["loss_seg"] = self.loss_decode(
            seg_logit,
            topk_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)

        loss["acc_seg"] = accuracy(seg_logit, topk_label)
        loss["acc_pix"] = pixel_recall(hist, topk_index)
        loss["acc_img"] = img_recall(hist, topk_index)
        loss["acc_img_accuracy"] = img_accuracy(hist, topk_index)
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
