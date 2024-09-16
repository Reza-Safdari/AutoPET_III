import torch
from torch import nn
from monai.networks.nets import UNet, SegResNetDS
from monai.networks.layers.factories import Norm


class AttentionUnet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            kernel_size=3,
            up_kernel_size=3,
            act="PRELU",
            dropout=0.2,
            bias=True
        )

        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output attention map for each channel
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_map = self.attention(x)
        x = self.unet(x * attention_map)
        return x


class AttentionSegResNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.segresnet = SegResNetDS(
            spatial_dims=3,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dsdepth=3,
            init_filters=16,
            in_channels=in_channels,
            out_channels=1,
        )

        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output attention map for each channel
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_map = self.attention(x)
        x = self.segresnet(x * attention_map)
        return x

