import os
import torch
import logging
import torch.utils.model_zoo as model_zoo

from ...utils.download import download

from ...config import cfg

from .registry import BACKBONE_REGISTRY

from .hrnet import HighResolutionNet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet50c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet50-25c4b509.pth',
    'resnet101c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet101-2a57e44d.pth',
    'resnet152c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet152-0d43d698.pth',
    'xception65': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/tf-xception65-270e81cf.pth',
    'mobilenet_v2': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/mobilenetV2-15498621.pth',
    "hrnet_w18_small_v1": "https://dl.dropboxusercontent.com/s/do1we7sfxowkm2w/hrnet_w18_small_model_v1.pth?dl=0",
    "hrnet_w18_small_v2": "https://dl.dropboxusercontent.com/s/pz3vdp8jg0ffkbr/hrnet_w18_small_model_v2.pth?dl=0",
    "hrnet_w18": "https://dl.dropboxusercontent.com/s/cewi4owfrw00oza/hrnetv2_w18_imagenet_pretrained.pth?dl=0",
    "hrnet_w30": "https://dl.dropboxusercontent.com/s/r1rafhhw1hfhpgl/hrnetv2_w30_imagenet_pretrained.pth?dl=0",
    "hrnet_w32": "https://dl.dropboxusercontent.com/s/7bu7mku682von0f/hrnetv2_w32_imagenet_pretrained.pth?dl=0",
    "hrnet_w48": "https://dl.dropboxusercontent.com/s/54s8dav0p32q9hx/hrnetv2_w48_imagenet_pretrained.pth?dl=0"
}

def load_backbone_pretrained(model, backbone):
    if cfg.PHASE == 'train' and cfg.TRAIN.BACKBONE_PRETRAINED and (not cfg.TRAIN.PRETRAINED_MODEL_PATH):
        if os.path.isfile(cfg.TRAIN.BACKBONE_PRETRAINED_PATH):
            logging.info('Load backbone pretrained model from {}'.format(
                cfg.TRAIN.BACKBONE_PRETRAINED_PATH
            ))
            msg = model.load_state_dict(torch.load(cfg.TRAIN.BACKBONE_PRETRAINED_PATH), strict=False)
            logging.info(msg)
        elif backbone not in model_urls:
            logging.info('{} has no pretrained model'.format(backbone))
            return
        else:
            logging.info('load backbone pretrained model from url..')
            try:
                if isinstance(model, HighResolutionNet):
                    weights_path = os.path.join(torch.hub._get_torch_home(), 'checkpoints')
                    weights_path = download(model_urls[backbone], path=weights_path)

                    msg = model.init_weights(pretrained=weights_path)
                else:    
                    msg = model.load_state_dict(model_zoo.load_url(model_urls[backbone]), strict=False)
            except Exception as e:
                logging.warning(e)
                logging.info('Use torch download failed, try custom method!')
                
                weights_path = os.path.join(torch.hub._get_torch_home(), 'checkpoints')
                weights_path = download(model_urls[backbone], path=weights_path)

                if isinstance(model, HighResolutionNet):
                    msg = model.init_weights(pretrained=weights_path)
                else:   
                    msg = model.load_state_dict(torch.load(weights_path), strict=False)
            logging.info(msg)


def get_segmentation_backbone(backbone, norm_layer=torch.nn.BatchNorm2d):
    """
    Built the backbone model, defined by `cfg.MODEL.BACKBONE`.
    """
    model = BACKBONE_REGISTRY.get(backbone)(norm_layer)
    load_backbone_pretrained(model, backbone)
    return model

