# this code is based on https://github.com/HRNet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01



architecture_config = {
    "hrnet_w18_small_v1": {
        "FINAL_CONV_KERNEL": 1
    },
    "hrnet_w18_small_v2": {
        "FINAL_CONV_KERNEL": 1
    },
    "hrnet_w18": {
        "FINAL_CONV_KERNEL": 1
    },
    "hrnet_w30": {
        "FINAL_CONV_KERNEL": 1
    },
    "hrnet_w32": {
        "FINAL_CONV_KERNEL": 1
    },
    "hrnet_w48": {
        "FINAL_CONV_KERNEL": 1
    }
}

class HRNet(SegBaseModel):
    def __init__(self, backbone_name, nclass, pretrained=True):
        global architecture_config
        config = architecture_config[backbone_name]

        self.backbone_name = backbone_name
        self.nclass = nclass
        super(HRNet, self).__init__(backbone_name=self.backbone_name, nclass=self.nclass, pretrained=pretrained)

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=self.backbone.last_inp_channels,
                out_channels=self.backbone.last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(self.backbone.last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.backbone.last_inp_channels,
                out_channels=nclass,
                kernel_size=config["FINAL_CONV_KERNEL"],
                stride=1,
                padding=1 if config["FINAL_CONV_KERNEL"] == 3 else 0)
        )
        
    def forward(self, x):
        ori_height, ori_width =x.shape[2],x.shape[3]
        x = self.backbone(x)

        x = self.head(x)
        x = F.interpolate(x, size=(ori_height, ori_width),
                        mode='bilinear') 
        return x



# def HRNet(nclass=2, backbone_name="hrnet_w18", pretrained=True):
#     cfg = architecture_config[backbone_name]
#     model = HRNetModel(cfg, nclass)
#     if pretrained:
#         model.backbone.init_weights(architecture_weights_path[backbone_name])
#     return model
