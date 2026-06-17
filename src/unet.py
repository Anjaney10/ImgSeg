"""U-Net model architecture for image segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity(),
        )
    
    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downsampling block with max pooling."""
    
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_p),
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling block with bilinear interpolation."""
    
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_p=0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, dropout_p)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels, dropout_p)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 to match x2 size if necessary
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for image segmentation."""
    
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        init_features=64,
        depth=4,
        bilinear=True,
        dropout_p=0.0,
    ):
        """
        Initialize U-Net model.
        
        Args:
            in_channels: Number of input channels (e.g., 3 for RGB)
            out_channels: Number of output channels (e.g., 1 for binary segmentation)
            init_features: Number of features in first conv layer
            depth: Number of downsampling levels (4 or 5)
            bilinear: Use bilinear upsampling instead of transposed conv
            dropout_p: Dropout probability for regularization
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.depth = depth
        
        # Input convolution
        self.inc = DoubleConv(in_channels, init_features, dropout_p)
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        for i in range(depth):
            in_ch = init_features * (2 ** i)
            out_ch = init_features * (2 ** (i + 1))
            self.encoder.append(DownBlock(in_ch, out_ch, dropout_p))
        
        # Bottleneck
        bottleneck_in = init_features * (2 ** depth)
        self.bottleneck = DoubleConv(bottleneck_in, bottleneck_in, dropout_p)
        
        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        for i in reversed(range(depth)):
            in_ch = init_features * (2 ** (i + 2))
            out_ch = init_features * (2 ** i)
            self.decoder.append(UpBlock(in_ch, out_ch, bilinear, dropout_p))
        
        # Output convolution
        self.outc = nn.Conv2d(init_features, out_channels, kernel_size=1)
    
    def forward(self, x):
        """Forward pass."""
        # Initial convolution
        x1 = self.inc(x)
        
        # Encoder path with skip connections
        skip_connections = [x1]
        x = x1
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(skip_connections[-1])
        
        # Decoder path
        for i, up in enumerate(self.decoder):
            skip = skip_connections[-(i + 2)]
            x = up(x, skip)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def get_num_parameters(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_unet(
    in_channels=3,
    out_channels=1,
    init_features=64,
    depth=4,
    pretrained=False,
):
    """
    Create U-Net model.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        init_features: Number of features in first layer
        depth: Number of downsampling blocks
        pretrained: Not used (included for API compatibility)
    
    Returns:
        UNet model instance
    """
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        depth=depth,
    )
    return model
