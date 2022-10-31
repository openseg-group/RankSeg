# Ported from https://github.com/PkuRainBow/HRT-Cls/blob/91d94a94585624e68d8d257dc850633606e94b58/models/modules/multihead_attention.py#L38

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.init import trunc_normal_
from torch._C import _infer_size, _add_docstr
from torch.nn import _reduction as _Reduction
from torch.nn.modules import utils
from torch.nn.modules.utils import _single, _pair, _triple, _list_with_default
from torch.nn import grad  # noqa: F401
from torch import _VF
from torch._jit_internal import boolean_dispatch, List, Optional, _overload, Tuple

from mmseg.ops import resize

# from torch.overrides import has_torch_function, handle_torch_function

from typing import Optional, Tuple

import functools


class RPE(nn.Module):
    def __init__(self, num_heads, q_window_size, kv_window_size, with_rpe):
        super().__init__()
        self.window_size = q_window_size
        self.kv_window_size = kv_window_size
        self.with_rpe = with_rpe

        if self.with_rpe:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (self.window_size + self.kv_window_size - 1)
                    * (self.window_size + self.kv_window_size - 1),
                    num_heads,
                )
            )
            coords_flatten = self._get_coord(
                self.window_size, self.window_size
            )  # 2, Wh*Ww
            coords_flatten_kv = self._get_coord(
                self.kv_window_size, self.kv_window_size
            )  # 2, Wh*Ww
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten_kv[:, None, :]
            )  # 2, Wh*Ww, Whk*Wwk
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # Wh*Ww, Whk*Wwk, 2
            relative_coords[:, :, 0] += self.kv_window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.kv_window_size - 1
            relative_coords[:, :, 0] *= self.window_size + self.kv_window_size - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Whq*Wwk
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    @staticmethod
    def _get_coord(h, w, device="cpu"):
        coords_h = torch.arange(h, device=device)
        coords_w = torch.arange(w, device=device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        return torch.flatten(coords, 1)  # 2, Wh*Ww

    def _get_relative_position_index(self, q_shape):
        device = self.relative_position_bias_table.device
        coords_flatten = self._get_coord(*q_shape, device)  # 2, Wh*Ww
        coords_flatten_kv = self._get_coord(
            self.kv_window_size, self.kv_window_size, device
        )  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten_kv[:, None, :]
        )  # 2, Wh*Ww, Whk*Wwk
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Whk*Wwk, 2
        relative_coords[:, :, 0] += self.kv_window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.kv_window_size - 1
        relative_coords[:, :, 0] *= q_shape[1] + self.kv_window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Whq*Wwk

        return relative_position_index

    def forward(self, q_shape):
        if self.with_rpe:
            target_index = self._get_relative_position_index(q_shape)
            target_size = (
                q_shape[0] + self.kv_window_size - 1,
                q_shape[1] + self.kv_window_size - 1,
            )
            rpb = self.relative_position_bias_table.view(
                1,
                (self.window_size + self.kv_window_size - 1),
                (self.window_size + self.kv_window_size - 1),
                -1,
            ).permute(0, 3, 1, 2)
            rpb = resize(rpb, size=target_size, mode="bicubic", align_corners=False)
            rpb = rpb.view(-1, target_size[0] * target_size[1]).permute(1, 0)
            rpb = rpb[target_index.view(-1)].view(
                q_shape[0] * q_shape[1],
                self.kv_window_size * self.kv_window_size,
                -1,
            )
            # add relative positional bias
            # relative_position_bias = self.relative_position_bias_table[
            #     self.relative_position_index.view(-1)
            # ].view(
            #     self.window_size * self.window_size,
            #     self.kv_window_size * self.kv_window_size,
            #     -1,
            # )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = rpb.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
        else:
            relative_position_bias = None

        return relative_position_bias


class MultiheadAttention(nn.Module):
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        with_rpe=True,
        q_window_size=None,
        kv_window_size=None,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.in_proj_bias = None
        self.in_proj_weight = None
        self.bias_k = self.bias_v = None
        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj_weight = None
        self.add_zero_attn = add_zero_attn

        self.rpe = RPE(num_heads, q_window_size, kv_window_size, with_rpe)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        q_shape=(),
    ):
        residual_attn = self.rpe(q_shape)

        if not self._qkv_same_embed_dim:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                out_dim=self.vdim,
                residual_attn=residual_attn,
            )
        else:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                out_dim=self.vdim,
                residual_attn=residual_attn,
            )

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        out_dim: Optional[Tensor] = None,
        residual_attn: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        tgt_len, bsz, embed_dim = query.size()
        key = query if key is None else key
        value = query if value is None else value

        assert embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        v_head_dim = out_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q = self.q_proj(query) * scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads, v_head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if add_zero_attn:
            src_len += 1
            k = torch.cat(
                [
                    k,
                    torch.zeros(
                        (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                    ),
                ],
                dim=1,
            )
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                    ),
                ],
                dim=1,
            )
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        """
        Attention weight for the invalid region is -inf
        """
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        if residual_attn is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights += residual_attn.unsqueeze(0)
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        """
        Reweight the attention map before softmax().
        attn_output_weights: (b*n_head, n, hw)
        """
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, v_head_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, out_dim)
        )
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output
