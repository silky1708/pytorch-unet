'''
A simple U-Net model in PyTorch
'''

import torch.nn as nn
import torch
import random

# helper modules
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))

# random crop and concat through skip connection
class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        assert x1.size()[2] < x2.size()[2], "input to Concat module: x1 expected to have smaller spatial dimension than x2"
        dim_small = x1.size()[2]
        dim_large = x2.size()[2]
        x = random.randint(0, dim_large-dim_small)
        y = random.randint(0, dim_large-dim_small)

        x2 = x2[:,:,x:x+dim_small, y:y+dim_small]
        return torch.cat([x1,x2], dim=1)

# UNet module put together
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1_1 = ConvBlock(in_channels, 64, 3)
        self.conv1_2 = ConvBlock(64, 64, 3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = ConvBlock(64, 128, 3)
        self.conv2_2 = ConvBlock(128, 128, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = ConvBlock(128, 256, 3)
        self.conv3_2 = ConvBlock(256, 256, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = ConvBlock(256, 512, 3)
        self.conv4_2 = ConvBlock(512, 512, 3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = ConvBlock(512, 1024, 3)
        self.conv5_2 = ConvBlock(1024, 1024, 3)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        self.conv6_1 = ConvBlock(1024, 512, 3)
        self.conv6_2 = ConvBlock(512, 512, 3)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)

        self.conv7_1 = ConvBlock(512, 256, 3)
        self.conv7_2 = ConvBlock(256, 256, 3)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)

        self.conv8_1 = ConvBlock(256, 128, 3)
        self.conv8_2 = ConvBlock(128, 128, 3)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)

        self.conv9_1 = ConvBlock(128, 64, 3)
        self.conv9_2 = ConvBlock(64, 64, 3)
        self.conv9_3 = ConvBlock(64, out_channels=out_channels, kernel_size=1)

        self.concat = Concat()

    def forward(self, x):
        x1 = self.conv1_2(self.conv1_1(x))

        x2 = self.pool1(x1)
        x2 = self.conv2_2(self.conv2_1(x2))

        x3 = self.pool2(x2)
        x3 = self.conv3_2(self.conv3_1(x3))

        x4 = self.pool3(x3)
        x4 = self.conv4_2(self.conv4_1(x4))

        x5 = self.pool4(x4)
        x5 = self.conv5_2(self.conv5_1(x5))
        x5 = self.up1(x5)

        x6 = self.concat(x5, x4)
        x6 = self.conv6_2(self.conv6_1(x6))
        x6 = self.up2(x6)

        x7 = self.concat(x6, x3)
        x7 = self.conv7_2(self.conv7_1(x7))
        x7 = self.up3(x7)

        x8 = self.concat(x7, x2)
        x8 = self.conv8_2(self.conv8_1(x8))
        x8 = self.up4(x8)

        x9 = self.concat(x8, x1)
        x9 = self.conv9_3(self.conv9_2(self.conv9_1(x9)))

        return x9

if __name__=="__main__":
    unet = UNet(3, 1)
    x = torch.randn((1,3,572,572), dtype=torch.float32)
    mask = unet(x)
    print(mask.size())
