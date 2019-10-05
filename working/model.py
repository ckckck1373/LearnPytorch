# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable





# # Simple model as shown in Figure. 4
class ResBlock(nn.Module):
    def __init__(self, nFeat, kernel_size, bn=False, bias=True, act=nn.ReLU(True)):
        super(ResBlock, self).__init__()
        modulesR = []
        modulesR.append(nn.Conv2d(in_channel=nFeat, out_channel=nFeat, kernel_size=kernel_size, padding=kernel_size //2, bias=bias))
        modulesR.append(act)
        modulesR.append(nn.Conv2d(in_channel=nFeat, out_channel=nFeat, kernel_size=kernel_size, padding=kernel_size //2, bias=bias))
        self.body = nn.Sequential(*modulesR)


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
class Upsampler(nn.Module):
    def __init__(self, scale, nFeat, act=False): 
        super(Upsampler, self).__init__()
        modulesU=[]
        modulesU.append(nn.Conv2d(nFeat, 4*nFeat , 3, bias=True, padding= kernel_size //2))
        modulesU.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modulesU)
   

    def forward(self, x):
        output=self.body(x)
        return output


# 9/27 還沒有run過
class ZebraSRNet(nn.Module):
    def __init__(self, nFeat=64, nResBlock=16, nChannel=3, scale=4, kernel_size=3:
        super(ZebraSRNet, self).__init__()
        # add the defination of layer here
        self.conv1=nn.Conv2d(nChannel, nFeat, 3, 1, 1)
        modules=[]
        for _ in range(nResBlock): # Why we need for loop here? ＝>有幾個resnet就append幾次
            modules.append(ResBlock(nFeat))
        self.body = nn.Sequential(*modules) # What is the *'s meaning => 因為sequential不接受列表 我們需要用*運算符號對其進行分解
        self.upsamp1 = upsampler(scale//2, nFeat)
        self.upsamp2 = upsampler(scale//2, nFeat)
        self.conv2 = nn.Ccnv2d(nFeat, nChannel, 3, 1, 1)


    def forward(self, x):
        # connect the layer together according to the Fig. 6 in the pdf 
        x = self.conv1(x)
        f_x = self.body(x)
        f_x = f_x + x
        f_x = self.upsamp1(f_x)
        output = self.conv2(f_x)

        return output


    # def forward(self, x):
    #     # connect the layer together according to the Fig. 6 in the pdf 
    #     x = self.Conv2d(3, 16, 3, 1, bias=True)
    #     x1 = self.body(x)
    #     x1 = x1 + x
    #     x2 = self.body(x1)
    #     x2 = x2 + x1 + x
    #     x3 = upsampler.forward(self,x2) # 可以這樣用其他class的function嗎?
    #     res = nn.Convd(16, 3, 3, padding=1, bias=True )
    #     return res






#%%



