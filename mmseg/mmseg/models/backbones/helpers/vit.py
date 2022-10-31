import torch
import numpy as np
from typing import List


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = (
        conv_weight.float()
    )  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError("Weight format not supported by conversion.")
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= 3 / float(in_chans)
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


@torch.no_grad()
def load_weights_from_HRT_Cls_format(checkpoint):
    def _fix_key(pattern: str, repl: str):
        import re

        nonlocal state_dict
        state_dict = {
            re.sub(pattern, repl, key): value for key, value in state_dict.items()
        }

    assert "model" in checkpoint
    state_dict = checkpoint["model"]

    _fix_key(r"^blocks", "layers")
    _fix_key(r"^norm", "ln1")
    _fix_key(r"(\d+\.)norm([12])", r"\1ln\2")
    _fix_key(r"attn\.qkv\.", "attn.attn.in_proj_")
    _fix_key(r"attn\.proj\.", "attn.attn.out_proj.")
    _fix_key(r"mlp\.fc1", "ffn.layers.0.0")
    _fix_key(r"mlp\.fc2", "ffn.layers.1")

    return state_dict


@torch.no_grad()
def load_weights_from_npz(model, checkpoint_path: str, prefix: str = ""):
    """Load weights from .npz checkpoints for official Google Brain Flax implementation"""

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    def _copy(module_path: str, weight: torch.Tensor):
        nonlocal state_dict
        state_dict[module_path] = weight

    state_dict = {}

    w = np.load(checkpoint_path)
    if not prefix and "opt/target/embedding/kernel" in w:
        prefix = "opt/target/"

    embed_conv_w = adapt_input_conv(
        model.patch_embed.projection.weight.shape[1],
        _n2p(w[f"{prefix}embedding/kernel"]),
    )
    _copy("patch_embed.projection.weight", embed_conv_w)
    _copy("patch_embed.projection.bias", _n2p(w[f"{prefix}embedding/bias"]))
    _copy("cls_token", _n2p(w[f"{prefix}cls"], t=False))
    pos_embed_w = _n2p(w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False)
    # if pos_embed_w.shape != model.pos_embed.shape:
    #     pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
    #         pos_embed_w,
    #         model.pos_embed,
    #         getattr(model, "num_tokens", 1),
    #         model.patch_embed.grid_size,
    #     )
    _copy("pos_embed", pos_embed_w)
    _copy("ln1.weight", _n2p(w[f"{prefix}Transformer/encoder_norm/scale"]))
    _copy("ln1.bias", _n2p(w[f"{prefix}Transformer/encoder_norm/bias"]))
    # if (
    #     isinstance(model.head, nn.Linear)
    #     and model.head.bias.shape[0] == w[f"{prefix}head/bias"].shape[-1]
    # ):
    #     _copy("head.weight", _n2p(w[f"{prefix}head/kernel"]))
    #     _copy("head.bias", _n2p(w[f"{prefix}head/bias"]))
    # if (
    #     isinstance(getattr(model.pre_logits, "fc", None), nn.Linear)
    #     and f"{prefix}pre_logits/bias" in w
    # ):
    #     _copy("pre_logits.fc.weight", _n2p(w[f"{prefix}pre_logits/kernel"]))
    #     _copy("pre_logits.fc.bias", _n2p(w[f"{prefix}pre_logits/bias"]))
    for i, block in enumerate(model.layers.children()):
        block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
        mha_prefix = block_prefix + "MultiHeadDotProductAttention_1/"
        _copy(f"layers.{i}.ln1.weight", _n2p(w[f"{block_prefix}LayerNorm_0/scale"]))
        _copy(f"layers.{i}.ln1.bias", _n2p(w[f"{block_prefix}LayerNorm_0/bias"]))
        _copy(
            f"layers.{i}.attn.attn.in_proj_weight",
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/kernel"], t=False).flatten(1).T
                    for n in ("query", "key", "value")
                ]
            ),
        )
        _copy(
            f"layers.{i}.attn.attn.in_proj_bias",
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/bias"], t=False).reshape(-1)
                    for n in ("query", "key", "value")
                ]
            ),
        )
        _copy(
            f"layers.{i}.attn.attn.out_proj.weight",
            _n2p(w[f"{mha_prefix}out/kernel"]).flatten(1),
        )
        _copy(f"layers.{i}.attn.attn.out_proj.bias", _n2p(w[f"{mha_prefix}out/bias"]))
        for r in range(2):
            rid = "0.0" if r == 0 else "1"
            _copy(
                f"layers.{i}.ffn.layers.{rid}.weight",
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/kernel"]),
            )
            # getattr(block.mlp, f"fc{r + 1}").weight.copy_(
            #     _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/kernel"])
            # )
            _copy(
                f"layers.{i}.ffn.layers.{rid}.bias",
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/bias"]),
            )
            # getattr(block.mlp, f"fc{r + 1}").bias.copy_(
            #     _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/bias"])
            # )
        _copy(f"layers.{i}.ln2.weight", _n2p(w[f"{block_prefix}LayerNorm_2/scale"]))
        _copy(f"layers.{i}.ln2.bias", _n2p(w[f"{block_prefix}LayerNorm_2/bias"]))

    return state_dict
