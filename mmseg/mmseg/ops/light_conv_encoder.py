import torch
import torch.nn as nn
import math


class LightConv_Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        img_size,
        use_bn=True,
    ):
        super(LightConv_Encoder, self).__init__()

        DH = self.DH = img_size[0] // (2 ** len(hidden_channels))
        DW = self.DW = img_size[1] // (2 ** len(hidden_channels))

        self.num_patches = DH * DW
        self.patches_resolution = [DH, DW]

        modules = []

        norm_layer = nn.BatchNorm2d if use_bn else nn.Identity

        for h_dim in hidden_channels:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    norm_layer(h_dim),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                # norm_layer(out_channels),
                # nn.ReLU(inplace=True),
            )
        )

        self.encoder = nn.Sequential(*modules)

    def encode(self, input):
        return self.encoder(input)

    def forward(self, input, **kwargs):
        encoding = self.encode(input)
        B, C, H, W = encoding.shape
        encoding = encoding.view(B, C, -1).permute(0, 2, 1)
        return encoding
