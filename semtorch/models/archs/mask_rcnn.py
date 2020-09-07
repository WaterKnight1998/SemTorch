from fastai.vision.all import *
from fastai.metrics import *
from fastai.learner import Metric
from fastai.torch_core import flatten_check
from fastai.data.all import *

# Result visualization

from ...utils.mask_rcnn_visualize import display_groundtruth_vs_pred

from copy import deepcopy

# DataBlock and DataLoader

class MaskRCNNDict(dict):
    
    @classmethod
    def create(cls, dictionary): 
        return cls(dict({x:dictionary[x] for x in dictionary.keys()}))
    
    def show(self, ctx=None, **kwargs): 
        dictionary = self
        
        boxes = dictionary["boxes"]
        labels = dictionary["labels"]
        masks = dictionary["masks"]
        
        result = masks
        return show_image(result, ctx=ctx, **kwargs)

def MaskRCNNBlock(): 
    return TransformBlock(type_tfms=MaskRCNNDict.create, batch_tfms=IntToFloatTensor)

def get_y_fn(x): 
    return Path(str(x).replace("Images","Labels").replace(".png",".tif"))

def get_bbox(o):
    label_path = get_y_fn(o)
    mask=PILMask.create(label_path)
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    
    return TensorBBox.create([xmin, ymin, xmax, ymax])
    
def get_bbox_label(o):
    
    return TensorCategory([1])

# New AvgMetricthat changes the input to the function
from fastai.learner import AvgMetric
def mk_metric(m):
    "Convert `m` to an `AvgMetric`, unless it's already a `Metric`"
    return m if isinstance(m, Metric) else AvgMetricMaskRCNN(m)


class AvgMetricMaskRCNN(AvgMetric):
    def __init__(self, func):  
        super().__init__(func)
    def accumulate(self, learn):
        bs = find_bs(learn.yb)
        self.total += to_detach(self.func(learn.pred, learn.yb))
        self.count += bs

# Learner

class MaskRCNNLearner(Learner):
    def __init__(self, dls, model, loss_func=None, opt_func=Adam, lr=defaults.lr, splitter=trainable_params, cbs=None,
                 metrics=None, path=None, model_dir='models', wd=None, wd_bn_bias=False, train_bn=True,
                 moms=(0.95,0.85,0.95)):

        # Pass a loss function to avoid fastai2 find one
        super().__init__(dls, model, nn.BCELoss(), opt_func, lr, splitter, cbs,
                 metrics, path, model_dir, wd, wd_bn_bias, train_bn,
                 moms)
      
    # Adjusting the AvgMetric for my Batch
    @property
    def metrics(self): return self._metrics
    @metrics.setter
    def metrics(self,v): self._metrics = L(v).map(mk_metric)
    
    def all_batches(self):
        self.n_iter = len(self.dl)
        for o in enumerate(self.dl): self.one_batch(*o)

    def _split(self, b):
        inputs = []
        targets = []
        for elem in b:
            inputs.append(elem[0])
            targets.append(elem[1])
        self.xb,self.yb = inputs,targets

    def _do_one_batch(self):
        self.model.train()
        loss_dict = self.model(self.xb,deepcopy(self.yb))
        # Once computed the losses we change into val
        if not self.training:
            self.model.eval()
            self.pred = self.model(self.xb)
        else:
            self.pred=loss_dict
        self('after_pred')
        loss = sum(loss for loss in loss_dict.values())
        self.loss = loss;   
        self('after_loss')
        if not self.training: return
        self('before_backward')
        self._backward()
        self('after_backward')
        self._step()
        self('after_step')
        self.opt.zero_grad()  

    def one_batch(self, i, b):
        self.iter = i
        self._split(b)
        self._with_events(self._do_one_batch, 'batch', CancelBatchException)


    # Expect a List[PILImage]
    def predict(self, item, rm_type_tfms=None, with_input=False):
        inputs = []
        for elem in item:
            # Int to float
            aux=image2tensor(elem).float().div_(255.)
            # Move to model device
            aux=aux.to(next(self.model.parameters()).device)
            inputs.append(aux)
        # Changing the model to Inference
        self.model.eval()

        # Predicting
        preds=self.model(inputs)

        return preds


    def show_results(self, ds_idx=1, dl=None, max_n=9, shuffle=True, **kwargs):
        if dl is None: dl = self.dls[ds_idx].new(shuffle=shuffle)
        b = dl.one_batch()

        image=b[0][0]

        # Predicting
        self.model.eval()
        preds=self.model([image])
        aux=b[0][1]

        maskPred=preds[0]["masks"]

        # If the model predict something show
        if maskPred.shape[0]>0:
            maskPred=maskPred[0]

            # Softmaxing result
            maskPred=maskPred>0.5
            maskPred=maskPred.type(torch.uint8)
            maskPred=maskPred.cuda()

            # Show
            original=dict(boxes=aux["boxes"],masks=aux["masks"],class_ids=aux["labels"],class_names=["Background","Tumor"])
            pred=dict(boxes=preds[0]["boxes"][0].unsqueeze(0),masks=maskPred,class_ids=TensorCategory([1]),class_names=["Background","Tumor"])
            display_groundtruth_vs_pred(image=image,original=original,pred=pred)

        else:
            print("The model didn't predicted anything")

# MIXED PRECISION

# ISSUE torch.cuda.amp https://github.com/pytorch/pytorch/issues/37735

# New DataBlock

from fastai.data.core import TfmdDL
class TfmdDLV2(TfmdDL):
    def __init__(self, dataset, bs=64, shuffle=False, num_workers=None, verbose=False, do_setup=True, **kwargs):
        super().__init__(dataset, bs, shuffle, num_workers, verbose, do_setup, **kwargs)

    def create_batch(self, b): 
        return b

# Transform
from fastai.vision.core import image2tensor
class IntToFloatTensorMaskRCNN(ItemTransform):
    "Transform image to float tensor, optionally dividing by 255 (e.g. for images)."
    def __init__(self, div=255., div_mask=1): store_attr()
    def encodes(self, o): 
        target={"boxes":o[1]["boxes"],
                "labels":o[1]["labels"],
                "masks":o[1]["masks"]
                }

        return (image2tensor(o[0]).float().div_(self.div),target)