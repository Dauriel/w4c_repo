import torch
from torch import nn
from models.unet_parts import OutConv
from models.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from models.layers import CBAM
from models.simvp_model import MidNet

class SmaAt_UNet(nn.Module):
    def __init__(self, in_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16, 
                 hid_T=256, N_S=4, N_T=8, mlp_ratio=8., drop=0.0, drop_path=0.1):
        super(SmaAt_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConvDS(self.in_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)

        self.mid = MidNet(512, hid_T, N_T,
            mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
                     
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1Att)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2Att)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3Att)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4Att)
        x5Att = self.cbam5(x5)

        x = self.mid(x5Att)

        x = self.up1(x, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)

        logits = self.outc(x)
        return logits
