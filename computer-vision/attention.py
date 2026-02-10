import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    """Coordinate Attention module.
    Focuses on spatial location and channel information.
    Implementation adapted to accept a 4D tensor (N,C,H,W) and return same shape.
    """
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Encode along height and width
        x_h = self.pool_h(x)               # (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (N, C, W, 1) -> permute -> (N, C, W, 1)

        # Concatenate along spatial dim
        y = torch.cat([x_h, x_w], dim=2)   # (N, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out

