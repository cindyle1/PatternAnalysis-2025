#check model builds (based off "ConvNeXt for the 2020s")
#code is found in https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

import torch
import torch.nn as nn

class Block(nn.Module):
    """
    ConvNeXt block that stays entirely in NCHW:
      - depthwise conv (7x7)
      - channels_first LayerNorm
      - 1x1 pwconv1 (expand 4x)
      - GELU
      - 1x1 pwconv2 (project back)
      - gamma (layer scale)
      - DropPath
      - residual
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise
        self.norm   = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        if layer_scale_init_value is not None and layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)))
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)                 # [B, C, H, W]
        x = self.norm(x)                   # channels_first LN
        x = self.pwconv1(x)                # [B, 4C, H, W]
        x = self.act(x)
        x = self.pwconv2(x)                # [B, C, H, W]
        if self.gamma is not None:
            # layer scale is per-channel; reshape for NCHW
            x = (self.gamma[:, None, None] * x)
        x = self.drop_path(x) + shortcut
        return x
    

class ConvNeXt(nn.Module):
    """
    Minimal ConvNeXt-Tiny style model tailored for grayscale images and 2-class output.

    Args:
        num_classes: number of output classes (default 2)
        in_chans:    input channels (default 1 for grayscale)
        depths:      number of blocks per stage
        dims:        channel widths per stage
        drop_path_rate: stochastic depth rate across blocks
        layer_scale_init_value: initial gamma value in blocks
        pretrained:  kept for API compatibility; no weights loaded here
    """
    def __init__(
        self,
        num_classes: int = 2,
        in_chans: int = 1,
        depths = (3, 3, 9, 3),
        dims   = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        pretrained: bool = False,
    ):
        super().__init__()
        assert len(depths) == 4 and len(dims) == 4, "depths and dims must have length 4"

        self.downsample_layers = nn.ModuleList()
        # Stem: conv stride 4 to quickly reduce resolution, then channels_first LN
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        # 3 downsample transitions between stages (2x each)
        for i in range(3):
            down = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(down)

        # Stochastic depth schedule
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        dp_iter = 0

        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(Block(dims[i], drop_path=dp_rates[dp_iter], layer_scale_init_value=layer_scale_init_value))
                dp_iter += 1
            self.stages.append(nn.Sequential(*blocks))

        # Head: global average pool (mean over H,W), then LayerNorm over C and Linear to classes
        self.norm_head = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head      = nn.Linear(dims[-1], num_classes)

        # Kaiming init is fine here
        self.apply(self._init_weights)

        if pretrained:
            # Placeholder to match your train.py flag; no external weights provided.
            print("[ConvNeXt] 'pretrained=True' requested, but no weights are bundled. Proceeding with random init.")

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, C, H, W]  (NCHW)
        for i in range(4):
            x = self.downsample_layers[i](x)  # stem or downsample
            x = self.stages[i](x)

        # global average pool over H,W -> [B, C]
        x = x.mean(dim=[2, 3])
        x = self.norm_head(x)                # LN over last dim (C)
        x = self.head(x)                     # logits [B, num_classes]
        return x

class LayerNorm(nn.Module):
    """
    LayerNorm that supports both channels_last (default) and channels_first.
    For channels_first, it normalizes over channel dim (C) with broadcasting.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        assert data_format in ("channels_last", "channels_first")
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            # x: [B, H, W, C] or [B, C]
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            # x: [B, C, H, W]  -> normalize per channel across H,W
            # mean/var over (H, W)
            mean = x.mean(dim=(2, 3), keepdim=True)
            var  = x.var (dim=(2, 3), keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
            # reshape weight/bias for broadcasting
            w = self.weight[:, None, None]
            b = self.bias  [:, None, None]
            return x * w + b


class DropPath(nn.Module):
    """Stochastic Depth per sample (when training only)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor