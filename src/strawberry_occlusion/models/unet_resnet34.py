"""ResNet34-encoder U-Net segmentation model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34


class ConvBlock(nn.Module):
    """Two convolution layers used in the U-Net decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample decoder features, concatenate a skip tensor, and refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.conv(torch.cat([x, skip], dim=1))


class ResNet34UNet(nn.Module):
    """U-Net with a ResNet34 encoder and class-logit segmentation head."""

    def __init__(
        self,
        num_classes: int,
        *,
        encoder_weights: ResNet34_Weights | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1")

        encoder = resnet34(weights=encoder_weights)
        self.input_stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4

        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        input_size = image.shape[-2:]

        stem = self.input_stem(image)
        pooled = self.maxpool(stem)
        enc1 = self.encoder1(pooled)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec4 = self.decoder4(enc4, enc3)
        dec3 = self.decoder3(dec4, enc2)
        dec2 = self.decoder2(dec3, enc1)
        dec1 = self.decoder1(dec2, stem)

        logits = self.classifier(dec1)
        return F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
