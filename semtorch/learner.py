import json

import os

from fastai.optimizer import Adam
from fastai.torch_core import trainable_params
from fastai.learner import defaults, Learner

from fastai.vision import models

# UNET
from fastai.vision.learner import unet_learner

# DeepLabV3+
from .models.archs.deeplabv3_plus import DeepLabV3Plus

# HRNET
from .models.archs.hrnet_seg import HRNet

# Mask-RCNN
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from .models.archs import mask_rcnn

# U²-Net
from .models.archs import u2net


current_folder = os.path.dirname(os.path.realpath(__file__))

architectures_config = json.load(open(f"{current_folder}{os.sep}architectures_config.json"))


class BackboneNotSupportedError(Exception):
    def __init__(self, backbone, supported_backbones):
        self.message = f"The backbone {backbone} is not supported. Supported ones are: {supported_backbones}"


class ArchitectureConfigNotSupportedError(Exception):
    def __init__(self, segmentation_type, number_classes):
        self.message = f"Segmentation Type: {segmentation_type} and Number of classes: {number_classes} is currently not supported"


def check_architecture_configuration(number_classes, segmentation_type, architecture_name, backbone_name):
    architecture_config = architectures_config[architecture_name]
    if not backbone_name in architecture_config["backbones"]:
        raise BackboneNotSupportedError(backbone_name, architecture_config["backbones"])

    bad_config = True
    print(architecture_config)
    print(segmentation_type, number_classes)
    for supported_config in architecture_config["supported_configs"]:
        if (segmentation_type in supported_config["segmentation_type"] and number_classes in supported_config["number_of_classes"]):
            bad_config = False
    if bad_config:
        raise ArchitectureConfigNotSupportedError(segmentation_type, number_classes)


unet_backbone_name = {
    "resnet18" : models.resnet18, 
    "resnet34" : models.resnet34, 
    "resnet50" : models.resnet50,
    "resnet101" : models.resnet101,
    "resnet152" : models.resnet152,
    "xresnet18" : models.xresnet.xresnet18,
    "xresnet34" : models.xresnet.xresnet34,
    "xresnet50" : models.xresnet.xresnet50,
    "xresnet101" : models.xresnet.xresnet101,
    "xresnet152" : models.xresnet.xresnet152,
    "squeezenet1_0" : models.squeezenet1_0,
    "squeezenet1_1" : models.squeezenet1_1,
    "densenet121" :  models.densenet121,
    "densenet169" :  models.densenet169,
    "densenet201" :  models.densenet201,
    "densenet161" :  models.densenet161,
    "vgg11_bn" :  models.vgg11_bn,
    "vgg13_bn" :  models.vgg13_bn,
    "vgg16_bn" :  models.vgg16_bn,
    "vgg19_bn" :  models.vgg19_bn,
    "alexnet" :  models.alexnet
}

def get_segmentation_learner(dls, number_classes, segmentation_type, architecture_name, backbone_name,
                             loss_func=None, opt_func=Adam, lr=defaults.lr, splitter=trainable_params, 
                             cbs=None, pretrained=True, normalize=True, image_size=None, metrics=None, 
                             path=None, model_dir='models', wd=None, wd_bn_bias=False, train_bn=True,
                             moms=(0.95,0.85,0.95), n_in=3):
    """This function return a learner for the provided architecture and backbone

    Parameters:
    dls (DataLoader): the dataloader to use with the learner
    number_classes (int): the number of clases in the project. It should be >=2
    segmentation_type (str): just `Semantic Segmentation` accepted for now 
    architecture_name (str): name of the architecture. The following ones are supported: `unet`, `deeplabv3+`, `hrnet`, `maskrcnn` and `u2^net`
    backbone_name (str): name of the backbone
    loss_func (): loss function.
    opt_func (): opt function.
    lr (): learning rates
    splitter (): splitter function for freazing the learner
    cbs (List[cb]): list of callbacks
    pretrained (bool): it defines if a trained backbone is needed
    normalize (bool): 
    image_size (int): REQUIRED for MaskRCNN. It indicates the desired size of the image.
    metrics (List[metric]): list of metrics
    path (): path parameter
    model_dir (str): the path in which save models
    wd (float): wieght decay
    wd_bn_bias (bool):
    train_bn (bool):
    moms (Tuple(float)): tuple of different momentuns
    n_in (int): Number of input channels
                    

    Returns:
    learner: value containing the learner object

    """

    
    number_classes_name = ""
    if number_classes == 2:
        number_classes_name = "binary"
    elif number_classes>2:
        number_classes_name = "multiple"
    else:
        raise Exception("The number of classes must be >=2")

    if n_in != 3 and architecture_name not in ['unet', 'u2^net']:
        raise Exception("More than 3 input channels is currently only supported for unet and u^2net")

    if n_in != 3 and pretrained == True:
        raise Exception("Error: Can only use pretrained backbones with three input channels")


    check_architecture_configuration(number_classes=number_classes_name, segmentation_type=segmentation_type, 
                                     architecture_name=architecture_name, backbone_name=backbone_name)


    learner = None

    if architecture_name == "unet":
        
        # TODO -> Revisar arch
        learner = unet_learner(dls=dls, arch=unet_backbone_name[backbone_name], metrics=metrics, wd=wd,
                               loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                               path=path, model_dir=model_dir, wd_bn_bias=wd_bn_bias, train_bn=train_bn,
                               pretrained=pretrained, normalize=normalize, moms=moms, n_in=n_in)
    
    
    elif architecture_name == "deeplabv3+":
        
        model = DeepLabV3Plus(backbone_name=backbone_name, nclass=number_classes, pretrained=pretrained)
        learner = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, 
                          cbs=cbs, metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn)
    
    
    elif architecture_name == "hrnet":

        model = HRNet(nclass=number_classes, backbone_name=backbone_name, pretrained=pretrained)
        learner = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                          metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn)


    elif architecture_name == "maskrcnn":
        if image_size is None:
            raise Exception("MaskRCNN need to define image_size. This values are for reescaling the image")

        model = maskrcnn_resnet50_fpn(num_classes=number_classes, min_size=image_size, max_size=image_size)
        learner = mask_rcnn.MaskRCNNLearner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                                            metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn)


    elif architecture_name == "u2^net":
        model = None
        if backbone_name == "small":
            model = u2net.U2NETP(n_in,1)
        elif backbone_name == "normal":
            model = u2net.U2NET(n_in,1)

        learner = u2net.USquaredNetLearner(dls=dls, model=model, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                                           metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn)

    return learner