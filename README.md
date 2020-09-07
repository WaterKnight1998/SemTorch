# SemTorch

This repository contains different deep learning architectures definitions that can be applied to image segmentation. 

All the architectures are implemented in [PyTorch](https://pytorch.org/) and can been trained easily with [FastAI 2](https://github.com/fastai/fastaihttps://github.com/fastai/fastai). 

In [Deep-Tumour-Spheroid repository](https://github.com/WaterKnight1998/Deep-Tumour-Spheroid) can be found and example of how to apply it with a custom dataset, in that case brain tumours images are used.

These architectures are classified as:

* **Semantic Segmentation:** each pixel of an image is linked to a class label.
![Semantic Segmentation](https://raw.githubusercontent.com/WaterKnight1998/SemTorch/develop/readme_images/semantic_segmentation.png)
* **Instance Segmentation:** is similar to semantic segmentation, but goes a bit deeper, it identifies , for each pixel, the object instance it belongs to.
![Instance Segmentation](https://raw.githubusercontent.com/WaterKnight1998/SemTorch/develop/readme_images/instance_segmentation.png)
* **Salient Object Detection (Binary clases only):** detection of the most noticeable/important object in an image.
![Salient Object Detection](https://raw.githubusercontent.com/WaterKnight1998/SemTorch/develop/readme_images/salient_object_detection.png)

## 🚀 Getting Started

To start using this package, install it using `pip`:

For example, for installing it in Ubuntu use:
```bash
pip3 install SemTorch
```

## 👩‍💻 Usage
This package creates an abstract API to access a segmentation model of different architectures. This method returns a FastAI 2 learner that can be combined with all the fastai's functionalities.

```
# SemTorch
from semtorch import get_segmentation_learner

learn = get_segmentation_learner(dls=dls, number_classes=2, segmentation_type="Semantic Segmentation",
                                 architecture_name="deeplabv3+", backbone_name="resnet50", 
                                 metrics=[tumour, Dice(), JaccardCoeff()],wd=1e-2,
                                 splitter=segmentron_splitter).to_fp16()
```

You can find a deeper example in [Deep-Tumour-Spheroid repository](https://github.com/WaterKnight1998/Deep-Tumour-Spheroid/tree/feature/notebooks), in this repo the package is used for the segmentation of brain tumours.

```
def get_segmentation_learner(dls, number_classes, segmentation_type, architecture_name, backbone_name,
                             loss_func=None, opt_func=Adam, lr=defaults.lr, splitter=trainable_params, 
                             cbs=None, pretrained=True, normalize=True, image_size=None, metrics=None, 
                             path=None, model_dir='models', wd=None, wd_bn_bias=False, train_bn=True,
                             moms=(0.95,0.85,0.95)):
```

This function return a learner for the provided architecture and backbone

### **Parameters:**

* **dls (DataLoader):** the dataloader to use with the learner
* **number_classes (int):** the number of clases in the project. It should be >=2
* **segmentation_type (str):** just `Semantic Segmentation` accepted for now 
* **architecture_name (str):** name of the architecture. The following ones are supported: `unet`, `deeplabv3+`, `hrnet`, `maskrcnn` and `u2^net`
* **backbone_name (str):** name of the backbone
* **loss_func ():** loss function.
* **opt_func ():** opt function.
* **lr ():** learning rates
* **splitter ():** splitter function for freazing the learner
* **cbs (List[cb]):** list of callbacks
* **pretrained (bool):** it defines if a trained backbone is needed
* **normalize (bool):** if normalization  is applied
* **image_size (int):** REQUIRED for MaskRCNN. It indicates the desired size of the image.
* **metrics (List[metric]):** list of metrics
* **path ():** path parameter
* **model_dir (str):** the path in which save models
* **wd (float):** wieght decay
* **wd_bn_bias (bool):**
* **train_bn (bool):**
* **moms (Tuple(float)):** tuple of different momentuns

### **Returns:**

* **learner:** value containing the learner object

### **Supported configs**

| Architecture |                           supported config                           |                                                                                                                               backbones                                                                                                                              |
|--------------|:--------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| unet         |  `Semantic Segmentation`,`binary` `Semantic Segmentation`,`multiple` | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `xresnet18`, `xresnet34`, `xresnet50`, `xresnet101`, `xresnet152`, `squeezenet1_0`, `squeezenet1_1`, `densenet121`, `densenet169`, `densenet201`, `densenet161`, `vgg11_bn`, `vgg13_bn`, `vgg16_bn`, `vgg19_bn`, `alexnet` |
| deeplabv3+   |  `Semantic Segmentation`,`binary` `Semantic Segmentation`,`multiple` |                                                                  `resnet18`,  `resnet34`,  `resnet50`,  `resnet101`,  `resnet152`,  `resnet50c`,  `resnet101c`,  `resnet152c`,  `xception65`,  `mobilenet_v2`                                                                 |
| hrnet        | `Semantic Segmentation`,`binary`  `Semantic Segmentation`,`multiple` |                                                                              `hrnet_w18_small_model_v1`,  `hrnet_w18_small_model_v2`,  `hrnet_w18`,  `hrnet_w30`,  `hrnet_w32`,  `hrnet_w48`                                                                              |
| maskrcnn     |                   `Semantic Segmentation`,`binary`                   |                                                                                                                               `resnet50`                                                                                                                               |
| u2^net       |                   `Semantic Segmentation`,`binary`                   |                                                                                                                           `small`,  `normal`                                                                                                                          |

## 📩 Contact
📧 dvdlacallecastillo@gmail.com

💼 Linkedin [David Lacalle Castillo](https://es.linkedin.com/in/david-lacalle-castillo-5b6280173)