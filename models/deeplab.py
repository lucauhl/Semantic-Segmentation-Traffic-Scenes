import torch
import torch.nn as nn
import torch.nn.functional as F

"""
DeepLabV3+ Architecture implementation from scratch.
Reference: https://arxiv.org/abs/1802.02611
"""


class BasicConv(nn.Module):
    """Standard Convolution -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Module.
    Captures context at multiple scales using different dilation rates.
    """

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # 1x1 Conv
        self.branch1 = BasicConv(in_channels, out_channels, 1)

        # 3x3 Conv, Dilation 6
        self.branch2 = BasicConv(in_channels, out_channels, 3, padding=6, dilation=6)

        # 3x3 Conv, Dilation 12
        self.branch3 = BasicConv(in_channels, out_channels, 3, padding=12, dilation=12)

        # 3x3 Conv, Dilation 18
        self.branch4 = BasicConv(in_channels, out_channels, 3, padding=18, dilation=18)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.branch5_conv = BasicConv(in_channels, out_channels, 1)

        self.conv_cat = BasicConv(out_channels * 5, out_channels, 1)

    def forward(self, x):
        h, w = x.shape[2:]

        # Parallel branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # Global Pooling Branch (Image Level Features)
        b5 = self.global_avg_pool(x)
        b5 = self.branch5_conv(b5)
        b5 = F.interpolate(b5, size=(h, w), mode='bilinear', align_corners=True)

        # Concatenate and fuse
        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        out = self.conv_cat(out)
        return out


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.name = "deeplabv3plus"

        # --- Encoder (Simplified ResNet-like backbone) ---
        # strided convs and later dilated convs.

        # Initial Stem
        self.stem = nn.Sequential(
            BasicConv(3, 64, 3, padding=1, stride=2),  # Downsample /2
            BasicConv(64, 64, 3, padding=1),
            BasicConv(64, 128, 3, padding=1)
        )

        # Low-Level Features extraction point (Stride 4)
        self.layer1 = nn.Sequential(
            BasicConv(128, 256, 3, padding=1, stride=2),  # Downsample /4
            BasicConv(256, 256, 3, padding=1)
        )

        # Mid-Level (Stride 8)
        self.layer2 = nn.Sequential(
            BasicConv(256, 512, 3, padding=1, stride=2),  # Downsample /8
            BasicConv(512, 512, 3, padding=1)
        )

        # For simplicity in this scratch implementation, we do one last downsample to Stride 16
        self.layer3 = nn.Sequential(
            BasicConv(512, 1024, 3, padding=1, stride=2),  # Downsample /16
            BasicConv(1024, 1024, 3, padding=2, dilation=2)  # Dilated conv to expand field
        )

        # --- ASPP Module ---
        self.aspp = ASPP(in_channels=1024, out_channels=256)

        # --- Decoder ---
        # 1. Project low-level features (from layer1) to fewer channels
        self.low_level_project = BasicConv(256, 48, 1)

        # 2. Final Convolutions after concatenation
        self.decoder_conv = nn.Sequential(
            BasicConv(256 + 48, 256, 3, padding=1),
            BasicConv(256, 256, 3, padding=1)
        )

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_shape = x.shape[2:]

        # --- Encoder Pass ---
        x = self.stem(x)
        low_level_feat = self.layer1(x)  # Keep for decoder (Stride 4)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)  # Main output (Stride 16)

        # --- ASPP Pass ---
        x = self.aspp(x)

        # --- Decoder Pass ---
        # 1. Upsample High-Level features (ASPP output) by 4 to match Low-Level features
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)

        # 2. Project Low-Level features (1x1 Conv)
        low_level_feat = self.low_level_project(low_level_feat)

        # 3. Concatenate
        x = torch.cat([x, low_level_feat], dim=1)

        # 4. Refine
        x = self.decoder_conv(x)

        # 5. Final Prediction and Upsample to original image size
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


if __name__ == '__main__':
    input = torch.randn(1, 3, 512, 512)
    model = DeepLabV3Plus(num_classes=10)

    model.eval()

    output = model(input)
    print(f"Input size: {input.shape}")
    print(f"Output size: {output.shape}")