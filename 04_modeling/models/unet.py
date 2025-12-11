import torch
import torch.nn as nn

"""
U-Net architecture implementation with PyTorch based on Ronneberger et. al.
Reference: https://doi.org/10.48550/arXiv.1505.04597

Use padding=1 in the convolution blocks to preserve the spatial resolution, so no cropping is required
"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels): 
        super(UpConvBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels=in_channels + skip_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.name = "u-net"
        self.num_classes = num_classes

        ### Encoder
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.conv5 = ConvBlock(512, 1024)

        ### Decoder
        self.transposed_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv1 = UpConvBlock(512, 512, 512)
        self.transposed_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv2 = UpConvBlock(256, 256, 256)
        self.transposed_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv3 = UpConvBlock(128, 128, 128)
        self.transposed_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv4 = UpConvBlock(64, 64, 64)
        
        self.score = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        ### Encoder
        features1 = self.conv1(x)
        x = self.max_pool2d(features1)
        features2 = self.conv2(x)
        x = self.max_pool2d(features2)
        features3 = self.conv3(x)
        x = self.max_pool2d(features3)
        features4 = self.conv4(x)
        x = self.max_pool2d(features4)

        ### Bottleneck
        x = self.conv5(x)
    
        ### Decoder
        x = self.transposed_conv1(x)
        x = self.up_conv1(torch.cat([features4, x], dim=1))
        x = self.transposed_conv2(x)
        x = self.up_conv2(torch.cat([features3, x], dim=1))
        x = self.transposed_conv3(x)
        x = self.up_conv3(torch.cat([features2, x], dim=1))
        x = self.transposed_conv4(x)
        x = self.up_conv4(torch.cat([features1, x], dim=1))

        score = self.score(x)
        return score
    
if __name__ == '__main__':
    ### Sanity check
    # Forward pass with random image (BxCxHxW) -> Batchsize, Channels, Height, Width
    input = torch.randn(1, 3, 1024, 2048)

    model = UNet(10)

    # Output size should match input size in terms of (HxW)
    output = model(input)
    print(f"Output size: {output.shape}")