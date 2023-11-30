from torch import nn
from models.unet_parts import OutConv
from models.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from models.layers import CBAM

class SmaAt_UNet(nn.Module):
    def __init__(self, in_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
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
        
        self.down4 = DownDS(512, 1024, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024, reduction_ratio=reduction_ratio)
        
        self.down5 = DownDS(1024, 2048, kernels_per_layer=kernels_per_layer)
        self.cbam6 = CBAM(2048, reduction_ratio=reduction_ratio)
        
        self.down6 = DownDS(2048, 2048, kernels_per_layer=kernels_per_layer)
        self.cbam7 = CBAM(2048, reduction_ratio=reduction_ratio)
        
        factor = 2 if self.bilinear else 1
        
        self.up1 = UpDS(4096, 2048  // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(2048, 1024  // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(1024, 512  // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(512, 256  // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up5 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up6 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        
        x6 = self.down5(x5)
        x6Att = self.cbam6(x6)
        
        x7 = self.down6(x6)
        x7Att = self.cbam7(x7)
        
        x = self.up1(x7Att, x6Att)
        x = self.up2(x, x5Att)
        x = self.up3(x, x4Att)
        x = self.up4(x, x3Att)
        x = self.up5(x, x2Att)
        x = self.up6(x, x1Att)
        
        logits = self.outc(x)
        return logits