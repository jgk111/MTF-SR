import torch
import torch.nn as nn


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, num_filters=256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual


# 修改后的 EDSR 模型架构，加入了 PixelShuffle（Sub-Pixel Conv）
class EDSR(nn.Module):
    def __init__(self, scale=2, num_res_blocks=32, num_filters=256):
        super(EDSR, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)  # 输入通道为3，输出通道为256

        # 残差块堆叠
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)

        # Sub-Pixel Convolution层 (PixelShuffle) 用于上采样
        self.subpixel_conv = nn.Conv2d(num_filters, num_filters * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)  # 放大scale倍

        # 输出层
        self.output_conv = nn.Conv2d(num_filters, 1, kernel_size=3, padding=1)  # 最终输出单通道

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.res_blocks(x)
        x = self.conv2(x)

        # Sub-Pixel Convolution进行上采样
        x = self.subpixel_conv(x)
        x = self.pixel_shuffle(x)

        x = self.output_conv(x)
        return x
