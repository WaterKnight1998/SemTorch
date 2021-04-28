# This code is heavily based on https://github.com/LikeLy-Journey/SegmenTron/blob/master/segmentron/models/deeplabv3_plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from ..modules import _ConvBNReLU, SeparableConv2d, _ASPP, _FCNHead

__all__ = ['DeepLabV3Plus']


class DeepLabV3Plus(SegBaseModel):
    r"""DeepLabV3Plus
    Reference:
        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
    """

    def __init__(self, backbone_name, nclass, pretrained=True):
        self.backbone_name = backbone_name
        self.nclass = nclass
        super(DeepLabV3Plus, self).__init__(backbone_name=self.backbone_name,nclass=self.nclass, pretrained=pretrained)

        if self.backbone_name.startswith('mobilenet'):
            c1_channels = 24
            c4_channels = 320
        elif self.backbone_name in ['resnet34', 'resnet18'] :
            c1_channels = 64
            c4_channels = 512
        else:
            c1_channels = 256
            c4_channels = 2048
        self.head = _DeepLabHead(self.nclass, c1_channels=c1_channels, c4_channels=c4_channels)

        self.__setattr__('decoder', ['head'])

    def forward(self, x):
        size = x.size()[2:]
        c1, _, c3, c4 = self.backbone(x)

        x = self.head(c4, c1)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class _DeepLabHead(nn.Module):
    def __init__(self, nclass, c1_channels=256, c4_channels=2048, norm_layer=nn.BatchNorm2d, use_aspp=True, use_decoder=True):
        super(_DeepLabHead, self).__init__()
        self.use_aspp = use_aspp
        self.use_decoder = use_decoder
        last_channels = c4_channels
        if self.use_aspp:
            self.aspp = _ASPP(c4_channels, 256)
            last_channels = 256
        if self.use_decoder:
            self.c1_block = _ConvBNReLU(c1_channels, 48, 1, norm_layer=norm_layer)
            last_channels += 48
        self.block = nn.Sequential(
            SeparableConv2d(last_channels, 256, 3, norm_layer=norm_layer, relu_first=False),
            SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False),
            nn.Conv2d(256, nclass, 1))

    def forward(self, x, c1):
        size = c1.size()[2:]
        if self.use_aspp:
            x = self.aspp(x)
        if self.use_decoder:
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)
            c1 = self.c1_block(c1)
            return self.block(torch.cat([x, c1], dim=1))

        return self.block(x)
