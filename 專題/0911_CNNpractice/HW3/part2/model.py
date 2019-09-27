import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable





# Simple model as shown in Figure. 4
class ResBlock(nn.Module):
    def __init__(self, bias=True, act=nn.ReLU(True)):
        super(ResBlock, self).__init__()
        modules = []
        modules.append(nn.Conv2d(16, 16, 3, padding=1))
        modules.append(act)
        modules.append(nn.Conv2d(16, 16, 3, padding=1))
        self.body = nn.Sequential(*modules)
    def forward(self, x):
        res = self.body(x)
        res += x
        return res
        
# Simple model as shown in Figure. 4
class ResBlock(nn.Module):
    def __init__(self, nFeat, kernel_size=3, bn=False, bias=True, act=nn.ReLU(True)):
        super(ResBlock, self).__init__()
        modules = []
        modules.append(nn.Conv2d(
            nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias))
        modules.append(act)
        modules.append(nn.Conv2d(
            nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias))
        self.body = nn.Sequential(*modules)
    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# you can refer to ResBlock and  Figure 6 to construct the model definition of Upsampler and ZebraSRNet
#======================================================
class upsampler(nn.Module):
    def __init__(self, scale, nFeat, act=False):
        super(upsampler, self).__init__()

        # add the definition of layer here

    def forward(self, x):



        return x



class ZebraSRNet(nn.Module):
    def __init__(self, nFeat=64, nResBlock=16, nChannel=3, scale=4):
        super(ZebraSRNet, self).__init__()

        # add the definition of layer here

    def forward(self, x):
        # connect the layer together according to the Fig. 6 in the pdf 

        return output





