import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from ..builder import HEADS, build_loss
from .decode_head import BaseDecodeHead
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmseg.ops import resize
from ..losses.accuracy import accuracy, img_recall, pixel_recall, img_accuracy


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


@HEADS.register_module()
class MaskTransformerNNCEESSNetHead(BaseDecodeHead):

    def __init__(
        self,
        n_cls,
        topk_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
        d_ess=-1,
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
        self.topk_cls = topk_cls
        self.scale = d_model**-0.5
        if d_ess < 0:
            d_ess = d_model

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, dpr[i])
            for i in range(n_layers)
        ])

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale *
                                       torch.randn(d_model, d_ess))
        self.proj_classes = nn.Parameter(self.scale *
                                         torch.randn(d_model, d_ess))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)
        # self.x = torch.zeros(1955,171)
        # self.y = torch.zeros(1955,171)
        # self.k = 0

    def init_weights(self):
        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

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

    def forward(self, x):
        # x, targets = x
        targets = x.new_ones((1, 512, 512)).long()
        # targets = targets.squeeze(1)
        x = self._transform_inputs(x)
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
        masks = self.mask_norm(masks)
        B, HW, N = masks.size()
        masks = masks.view(B, H, W, N).permute(0, 3, 1, 2)

        if self.training:
            masks = resize(
                input=masks,
                size=targets.shape[1:],
                mode='bilinear',
                align_corners=self.align_corners)

            with torch.no_grad():
                _, topk_label = torch.topk(masks, self.topk_cls, dim=1)
                topk_label = topk_label.permute(1, 0, 2, 3)  # k B H W
                mask_tar = topk_label == targets
                topk_label[mask_tar] = topk_label[mask_tar] - 1
                topk_label = topk_label.permute(1, 0, 2, 3)  # B k H W
                index = torch.cat([targets.unsqueeze(1), topk_label],
                                  dim=1)  # B k+1 H W
                index[index == -1] = 0
                index[index == self.ignore_index] = 0

            topk_masks = torch.gather(masks, 1, index)
            return topk_masks
        # masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        else:
            with torch.no_grad():
                _, non_topk_label = torch.topk(
                    masks, self.n_cls - self.topk_cls, dim=1, largest=False)
                masks.scatter_(1, non_topk_label, -100)
            return masks

    @force_fp32(apply_to=('outputs', ))
    def losses(self, outputs, seg_label):
        """Compute segmentation loss."""
        pixel_pred = outputs
        loss = dict()

        seg_logit = resize(
            input=pixel_pred,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_label = self.ignore_index * (
            torch.logical_or(seg_label == -1, seg_label == self.ignore_index))

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
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