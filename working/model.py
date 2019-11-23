import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable





# Simple model as shown in Figure. 4
class ResBlock(nn.Module):
    def __init__(self, nFeat, kernel_size=3, bn=False, bias=True, act=nn.ReLU(True)):
        super(ResBlock, self).__init__()
        modules = []
        modules.append(nn.Conv2d(nFeat, nFeat, 3, padding=1, bias=bias))
        ### nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        modules.append(act)
        modules.append(nn.Conv2d(nFeat, nFeat, 3, padding=1, bias=bias))
        ### nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.body = nn.Sequential(*modules)
    def forward(self, x):
        res = self.body(x)
        res += x
        return res
        
# # Simple model as shown in Figure. 4
# class ResBlock(nn.Module):
#     def __init__(self, nFeat, kernel_size=3, bn=False, bias=True, act=nn.ReLU(True)):
#         super(ResBlock, self).__init__()
#         modules = []
#         modules.append(nn.Conv2d(
#             nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias))
#         modules.append(act)
#         modules.append(nn.Conv2d(
#             nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias))
#         self.body = nn.Sequential(*modules)
#     def forward(self, x):
#         res = self.body(x)
#         res += x
#         return res


# you can refer to ResBlock and  Figure 6 to construct the model definition of Upsampler and ZebraSRNet
#======================================================
#class upsampler(nn.Module):
#    def __init__(self, scale, nFeat, act=False):
#        super(upsampler, self).__init__()

        # add the definition of layer here

#    def forward(self, x):



#        return x
        
class upsampler(nn.Module):
    def __init__(self, scale, nFeat, act=False):
        super(upsampler, self).__init__()
        # add the definition of layer here
        self.conv = nn.Conv2d(nFeat, nFeat*4, 3, 1, 1)
        ### nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.shuffle = nn.PixelShuffle(scale)
        ### nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        return x



#class ZebraSRNet(nn.Module):
#    def __init__(self, nFeat=64, nResBlock=16, nChannel=3, scale=4):
#        super(ZebraSRNet, self).__init__()

        # add the definition of layer here

#    def forward(self, x):
        # connect the layer together according to the Fig. 6 in the pdf 

#        return output

class ZebraSRNet(nn.Module):
    def __init__(self, nFeat=64, nResBlock=8, nChannel=3, scale=2):
        super(ZebraSRNet, self).__init__()
        # add the definition of layer here
        self.conv1 = nn.Conv2d(nChannel, nFeat, 3, 1, 1)
        modules = []
        for _ in range(nResBlock):
            modules.append(ResBlock(nFeat))
        self.body = nn.Sequential(*modules)
        self.upsamp1 = upsampler(scale, nFeat)
        self.upsamp2 = upsampler(scale, nFeat)
        self.conv2 = nn.Conv2d(nFeat, nChannel, 3, 1, 1)

    def forward(self, x):
        # connect the layer together according to the Fig. 6 in the pdf 
        x = self.conv1(x)
        f_x = self.body(x)
        f_x = f_x + x
        f_x = self.upsamp1(f_x)
        f_x = self.upsamp2(f_x)
        output = self.conv2(f_x)

        return output



