#check model builds (based off "ConvNeXt for the 2020s")
#code is found in https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

import torch
import torch.nn as nn

class Block(nn.Module):
    """ConvNeXt block from the 022 paper"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv (7×7)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise expansion
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)  # pointwise projection
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))  # layer scale

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) → (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # back to (N, C, H, W)
        return x + shortcut
    

class ConvNeXt(nn.Module):
    """ConvNeXt-Tiny variant"""
    def __init__(self, in_chans=1, num_classes=2):
        super().__init__()
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]

        # Patchify stem (4×4 conv, stride 4)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)

        # Downsampling layers between stages
        for i in range(3):
            down = nn.Sequential(
                nn.LayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(down)

        # Stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[Block(dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        # Final norm and classification head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        # global average pooling
        x = x.mean([-2, -1])
        x = self.norm(x)
        return self.head(x)
