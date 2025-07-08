import torch
from torch import nn

class GlobalContext2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 1)  # [B, out_c, H, W]
        )

    def forward(self, x):  # x: [B, 3, H, W]
        return self.encoder(x)


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class UNet1D(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super().__init__()
        self.down1 = Conv1DBlock(in_channels, ngf, kernel_size=4, stride=2, padding=1)  # 1024 -> 512
        self.down2 = Conv1DBlock(ngf, ngf * 2, kernel_size=4, stride=2, padding=1)  # 512 -> 256
        self.down3 = Conv1DBlock(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1)  # 256 -> 128
        self.down4 = Conv1DBlock(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1)  # 128 -> 64

        self.bottleneck = Conv1DBlock(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)  # 64 -> 32

        self.up1 = nn.ConvTranspose1d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)  # 32 -> 64
        self.up2 = nn.ConvTranspose1d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1)  # 64 -> 128
        self.up3 = nn.ConvTranspose1d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1)  # 128 -> 256
        self.up4 = nn.ConvTranspose1d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1)  # 256 -> 512
        self.up5 = nn.ConvTranspose1d(ngf * 2, out_channels, kernel_size=4, stride=2, padding=1)  # 512 -> 1024

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x is a single line: (B, C, W)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        b = self.bottleneck(d4)

        u1 = self.relu(self.up1(b))
        u2 = self.relu(self.up2(torch.cat([u1, d4], dim=1)))
        u3 = self.relu(self.up3(torch.cat([u2, d3], dim=1)))
        u4 = self.relu(self.up4(torch.cat([u3, d2], dim=1)))
        u5 = self.tanh(self.up5(torch.cat([u4, d1], dim=1)))

        return u5