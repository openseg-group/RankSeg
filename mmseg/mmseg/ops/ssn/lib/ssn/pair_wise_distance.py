import os
import torch
from torch.utils.cpp_extension import load_inline, load
from .pair_wise_distance_cuda_source import source

# try:
#     from apex import amp
# except ImportError:
#     amp = None


def resolve(*parts):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), *parts)


try:
    import sys
    from pathlib import Path

    sys.path.insert(
        0,
        str(Path("/root") / ".cache" / "torch_extensions" / "pair_wise_distance_cuda"),
    )
    import pair_wise_distance_cuda

    # print("Prebuilt SSN extension detected")
except ModuleNotFoundError:
    sys.path.pop(0)

    print("compile cuda source of 'pair_wise_distance' function...")
    print(
        "NOTE: if you avoid this process, you make .cu file and compile"
        " it following https://pytorch.org/tutorials/advanced/cpp_extension.html"
    )

    pair_wise_distance_cuda = load(
        "pair_wise_distance_cuda",
        sources=[resolve("pair_wise_distance.cu")],
        verbose=True,
        # build_directory=resolve("build")
    )
    print("done")


def pairwise_dist(*args):
    return PairwiseDistFunction.apply(*args)


# if amp is not None:
#     pairwise_dist = amp.half_function(pairwise_dist)


class PairwiseDistFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        self,
        pixel_features,
        spixel_features,
        init_spixel_indices,
        num_spixels_width,
        num_spixels_height,
    ):
        self.num_spixels_width = num_spixels_width
        self.num_spixels_height = num_spixels_height
        output = pixel_features.new(
            pixel_features.shape[0], 9, pixel_features.shape[-1]
        ).zero_()
        self.save_for_backward(pixel_features, spixel_features, init_spixel_indices)

        # print(pixel_features.dtype, spixel_features.dtype, output.dtype,
        #   init_spixel_indices.dtype)
        return pair_wise_distance_cuda.forward(
            pixel_features.contiguous(),
            spixel_features.contiguous(),
            init_spixel_indices.contiguous(),
            output,
            self.num_spixels_width,
            self.num_spixels_height,
        )

    @staticmethod
    def backward(self, dist_matrix_grad):
        pixel_features, spixel_features, init_spixel_indices = self.saved_tensors

        pixel_features_grad = torch.zeros_like(pixel_features)
        spixel_features_grad = torch.zeros_like(spixel_features)

        pixel_features_grad, spixel_features_grad = pair_wise_distance_cuda.backward(
            dist_matrix_grad.contiguous(),
            pixel_features.contiguous(),
            spixel_features.contiguous(),
            init_spixel_indices.contiguous(),
            pixel_features_grad,
            spixel_features_grad,
            self.num_spixels_width,
            self.num_spixels_height,
        )
        return pixel_features_grad, spixel_features_grad, None, None, None
