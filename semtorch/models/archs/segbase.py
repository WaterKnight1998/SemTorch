# Base Model for Semantic Segmentation
# This code is heavily based on https://github.com/LikeLy-Journey/SegmenTron/blob/master/segmentron/models/segbase.py

import math
import numbers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import get_segmentation_backbone
from ..modules import get_norm
__all__ = ['SegBaseModel']


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation
    """
    def __init__(self, pretrained=True, backbone_name="resnet34", nclass=2,norm_layer='BN'):
        super(SegBaseModel, self).__init__()
        self.nclass = nclass
        self.norm_layer = get_norm(norm_layer)
        self.backbone = backbone_name
        if pretrained:
            self.get_backbone()

    def get_backbone(self):
        self.backbone = get_segmentation_backbone(self.backbone, self.norm_layer)

    def base_forward(self, x):
        """forwarding backbone network"""
        c1, c2, c3, c4 = self.backbone(x)
        return c1, c2, c3, c4

    def demo(self, x):
        pred = self.forward(x)
        return pred



def _resize_image(img, h, w):
    return F.interpolate(img, size=[h, w], mode='bilinear', align_corners=True)


def _pad_image(img, crop_size):
    b, c, h, w = img.shape
    assert(c == 3)
    padh = crop_size[0] - h if h < crop_size[0] else 0
    padw = crop_size[1] - w if w < crop_size[1] else 0
    if padh == 0 and padw == 0:
        return img
    img_pad = F.pad(img, (0, padh, 0, padw))

    return img_pad


def _crop_image(img, h0, h1, w0, w1):
    return img[:, :, h0:h1, w0:w1]


def _flip_image(img):
    assert(img.ndim == 4)
    return img.flip((3))


def _to_tuple(size):
    if isinstance(size, (list, tuple)):
        assert len(size), 'Expect eval crop size contains two element, ' \
                          'but received {}'.format(len(size))
        return tuple(size)
    elif isinstance(size, numbers.Number):
        return tuple((size, size))
    else:
        raise ValueError('Unsupport datatype: {}'.format(type(size)))
