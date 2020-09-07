import torch
import functools

BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
relu_inplace = True