import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

import numpy as np

from typing import List

from .registry import BACKBONE_REGISTRY

BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.norm_layer = norm_layer

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = self.norm_layer(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = self.norm_layer(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.norm_layer = norm_layer

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = self.norm_layer(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = self.norm_layer(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = self.norm_layer(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, norm_layer, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()

        self.norm_layer = norm_layer

        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(num_channels[branch_index] * block.expansion,
                                momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], self.norm_layer, stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], self.norm_layer))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        self.norm_layer(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3, 
                                                momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3,
                                                momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x: List[torch.Tensor]):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])

        x_fuse = []
        for i, fuse_layer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_layer[0](x[0])
            for j, fuse_sub_layer in enumerate(fuse_layer):
                if j == 0 or j > self.num_branches:
                    pass
                else:
                    if i == j:
                        y = y + x[j]
                    elif j > i:
                        width_output = x[i].shape[-1]
                        height_output = x[i].shape[-2]
                        y = y + F.interpolate(
                            fuse_sub_layer(x[j]),
                            size=[height_output, width_output],
                            mode='bilinear')
                    else:
                        y = y + fuse_sub_layer(x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, norm_layer, **kwargs):
        super(HighResolutionNet, self).__init__()

        self.norm_layer = norm_layer
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = self.norm_layer(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = self.norm_layer(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        
        self.stage1_cfg = config['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = config['STAGE2']

        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = config['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = config['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        self.last_inp_channels = np.int(np.sum(pre_stage_channels))

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        self.norm_layer(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        self.norm_layer(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, self.norm_layer, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, self.norm_layer))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      self.norm_layer,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        # return nn.Sequential(*modules), num_inchannels
        return nn.ModuleList(modules), num_inchannels

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for aux in self.transition1:
            if not isinstance(aux,nn.Identity):
                x_list.append(aux(x))
            else:
                x_list.append(x)
        #y_list = self.stage2(x_list)

        for aux in self.stage2:
            x_list = aux(x_list)
        y_list = x_list


        x_list = []
        for i, aux in enumerate(self.transition2):
            if not isinstance(aux,nn.Identity):
                x_list.append(aux(y_list[-1]))
            else:
                x_list.append(y_list[i])
        #y_list = self.stage3(x_list)
        for aux in self.stage3:
            x_list = aux(x_list)
        y_list = x_list

        x_list = []
        for i, aux in enumerate(self.transition3):
            if not isinstance(aux,nn.Identity):
                x_list.append(aux(y_list[-1]))
            else:
                x_list.append(y_list[i])
        #x = self.stage4(x_list)
        for aux in self.stage4:
            x_list = aux(x_list)
        x = x_list
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)

        # x = self.last_layer(x)

        # #UpSample
        # x = F.interpolate(x, size=(ori_height, ori_width),
        #                 mode='bilinear') 
        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            #for k, _ in pretrained_dict.items():
            #    logger.info(
            #        '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        return "HRNet backbone wieghts loaded"

backbone_config={
    "hrnet_w18_small_v1": {
        "STAGE1": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "BLOCK": "BOTTLENECK",
            "NUM_BLOCKS": [1],
            "NUM_CHANNELS": [32],
            "FUSE_METHOD": "SUM"
        },
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [2,2],
            "NUM_CHANNELS": [16,32],
            "FUSE_METHOD": "SUM"
        },
        "STAGE3": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 3,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [2,2,2],
            "NUM_CHANNELS": [16,32,64],
            "FUSE_METHOD": "SUM"
        },
        "STAGE4": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 4,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [2,2,2,2],
            "NUM_CHANNELS": [16,32,64,128],
            "FUSE_METHOD": "SUM"
        }
    },
    "hrnet_w18_small_v2": {
        "STAGE1": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "BLOCK": "BOTTLENECK",
            "NUM_BLOCKS": [2],
            "NUM_CHANNELS": [64],
            "FUSE_METHOD": "SUM"
        },
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [2,2],
            "NUM_CHANNELS": [18,36],
            "FUSE_METHOD": "SUM"
        },
        "STAGE3": {
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 3,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [2,2,2],
            "NUM_CHANNELS": [18,36,72],
            "FUSE_METHOD": "SUM"
        },
        "STAGE4": {
            "NUM_MODULES": 2,
            "NUM_BRANCHES": 4,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [2,2,2,2],
            "NUM_CHANNELS": [18, 36, 72, 144],
            "FUSE_METHOD": "SUM"
        }
    },
    "hrnet_w18": {
        "STAGE1": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "BLOCK": "BOTTLENECK",
            "NUM_BLOCKS": [4],
            "NUM_CHANNELS": [64],
            "FUSE_METHOD": "SUM"
        },
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4,4],
            "NUM_CHANNELS": [18,36],
            "FUSE_METHOD": "SUM"
        },
        "STAGE3": {
            "NUM_MODULES": 4,
            "NUM_BRANCHES": 3,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4,4,4],
            "NUM_CHANNELS": [18,36,72],
            "FUSE_METHOD": "SUM"
        },
        "STAGE4": {
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 4,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4,4,4,4],
            "NUM_CHANNELS": [18, 36, 72, 144],
            "FUSE_METHOD": "SUM"
        }
    },
    "hrnet_w30": {
        "STAGE1": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "BLOCK": "BOTTLENECK",
            "NUM_BLOCKS": [4],
            "NUM_CHANNELS": [64],
            "FUSE_METHOD": "SUM"
        },
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4,4],
            "NUM_CHANNELS": [30, 60],
            "FUSE_METHOD": "SUM"
        },
        "STAGE3": {
            "NUM_MODULES": 4,
            "NUM_BRANCHES": 3,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4],
            "NUM_CHANNELS": [30, 60, 120],
            "FUSE_METHOD": "SUM"
        },
        "STAGE4": {
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 4,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4, 4],
            "NUM_CHANNELS": [30, 60, 120, 240],
            "FUSE_METHOD": "SUM"
        }
    },
    "hrnet_w32": {
        "STAGE1": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "BLOCK": "BOTTLENECK",
            "NUM_BLOCKS": [4],
            "NUM_CHANNELS": [64],
            "FUSE_METHOD": "SUM"
        },
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4,4],
            "NUM_CHANNELS": [32, 64],
            "FUSE_METHOD": "SUM"
        },
        "STAGE3": {
            "NUM_MODULES": 4,
            "NUM_BRANCHES": 3,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4],
            "NUM_CHANNELS": [32, 64, 128],
            "FUSE_METHOD": "SUM"
        },
        "STAGE4": {
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 4,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4, 4],
            "NUM_CHANNELS": [32, 64, 128, 256],
            "FUSE_METHOD": "SUM"
        }
    },
    "hrnet_w48": {
        "STAGE1": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "BLOCK": "BOTTLENECK",
            "NUM_BLOCKS": [4],
            "NUM_CHANNELS": [64],
            "FUSE_METHOD": "SUM"
        },
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4,4],
            "NUM_CHANNELS": [48, 96],
            "FUSE_METHOD": "SUM"
        },
        "STAGE3": {
            "NUM_MODULES": 4,
            "NUM_BRANCHES": 3,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4],
            "NUM_CHANNELS": [48, 96, 192],
            "FUSE_METHOD": "SUM"
        },
        "STAGE4": {
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 4,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4, 4],
            "NUM_CHANNELS": [48, 96, 192, 384],
            "FUSE_METHOD": "SUM"
        }
    }
}

@BACKBONE_REGISTRY.register()
def hrnet_w18_small_v1(norm_layer=nn.BatchNorm2d):
    model = HighResolutionNet(config=backbone_config["hrnet_w18_small_v1"], norm_layer=norm_layer)
    return model

@BACKBONE_REGISTRY.register()
def hrnet_w18_small_v2(norm_layer=nn.BatchNorm2d):
    model = HighResolutionNet(config=backbone_config["hrnet_w18_small_v2"], norm_layer=norm_layer)
    return model

@BACKBONE_REGISTRY.register()
def hrnet_w18(norm_layer=nn.BatchNorm2d):
    model = HighResolutionNet(config=backbone_config["hrnet_w18"], norm_layer=norm_layer)
    return model

@BACKBONE_REGISTRY.register()
def hrnet_w30(norm_layer=nn.BatchNorm2d):
    model = HighResolutionNet(config=backbone_config["hrnet_w30"], norm_layer=norm_layer)
    return model

@BACKBONE_REGISTRY.register()
def hrnet_w32(norm_layer=nn.BatchNorm2d):
    model = HighResolutionNet(config=backbone_config["hrnet_w32"], norm_layer=norm_layer)
    return model

@BACKBONE_REGISTRY.register()
def hrnet_w48(norm_layer=nn.BatchNorm2d):
    model = HighResolutionNet(config=backbone_config["hrnet_w48"], norm_layer=norm_layer)
    return model
