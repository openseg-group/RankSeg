import torch
import torch.nn as nn
import math

# from .gSLICr.modules import DeNormalize, TransformColorSpace
from .ssn.lib.ssn.ssn import ssn_iter, sparse_ssn_iter


def get_aspect_ratio(x, y):
    gcd = math.gcd(x, y)
    return x // gcd, y // gcd


def reshape_as_aspect_ratio(x, ratio, channel_last=False):
    assert x.ndim == 3
    B, N, C = x.size()
    s = round(math.sqrt(N / (ratio[0] * ratio[1])))
    perm = (0, 1, 2) if channel_last else (0, 2, 1)
    return x.permute(*perm).view(B, C, s * ratio[0], s * ratio[1])


def convert_len_to_hw(x, ratio):
    assert x.ndim == 3
    B, N, C = x.size()
    s = round(math.sqrt(N / (ratio[0] * ratio[1])))
    return s * ratio[0], s * ratio[1]


def naive_unpool(f_regions, region_indices):
    _, _, C = f_regions.shape
    N, L = region_indices.shape
    index = region_indices.view(N, L, 1).expand(N, L, C)
    result = f_regions.gather(1, index)
    return result


class SSNIterations(nn.Module):
    def __init__(self, num_spixels, n_iters=5):
        super().__init__()
        if isinstance(num_spixels, tuple):
            assert len(num_spixels) == 2
        else:
            x = int(math.sqrt(num_spixels))
            assert x * x == num_spixels
            num_spixels = (x, x)
        self.num_spixels = num_spixels
        self.n_iters = n_iters

    def forward(self, f):
        func = ssn_iter if self.training else sparse_ssn_iter
        *_, spixel_features, hard_labels = func(f, self.num_spixels, self.n_iters)
        return spixel_features.permute(0, 2, 1), hard_labels

    def extra_repr(self):
        return f"num_spixels={self.num_spixels}, n_iters={self.n_iters}"
