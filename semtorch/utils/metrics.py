from fastai.learner import Metric
from fastai.torch_core import flatten_check

import torch

class DiceMaskRCNN(Metric):
    "MaskRCNN coefficient metric for binary target in segmentation"
    def __init__(self, axis=1): self.axis = axis
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, learn):
        for i in range(len(learn.pred)):
            pred=learn.pred[i]["masks"]

            targ=learn.yb[i]["masks"].squeeze(1)
            # In case not predicted mask. Set all pixels to background
            if pred.shape[0]>0:
                pred=pred[0] 
            else:
                pred=torch.zeros(targ.shape).to(targ.device)
            pred,targ = flatten_check(pred, targ)
            self.inter += (pred*targ).float().sum().item()
            self.union += (pred+targ).float().sum().item()
        
    @property
    def value(self): return 2. * self.inter/self.union if self.union > 0 else None

# Cell
class JaccardCoeffMaskRCNN(DiceMaskRCNN):
    "Implemetation of the jaccard coefficient that is lighter in RAM"
    @property
    def value(self): return self.inter/(self.union-self.inter) if self.union > 0 else None

# Metric
class DiceUSquaredNet(Metric):
    "Dice coefficient metric for binary target in segmentation"
    def __init__(self, axis=1): self.axis = axis
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, learn):
        pred,targ = flatten_check(learn.pred, learn.y)
        self.inter += (pred*targ).float().sum().item()
        self.union += (pred+targ).float().sum().item()

    @property
    def value(self): return 2. * self.inter/self.union if self.union > 0 else None

# Cell
class JaccardCoeffUSquaredNet(DiceUSquaredNet):
    "Implemetation of the jaccard coefficient that is lighter in RAM"
    @property
    def value(self): return self.inter/(self.union-self.inter) if self.union > 0 else None