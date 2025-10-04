from ops import *
import torch
import torch.nn.functional as F
import cv2
import numpy as np


class CGBs(nn.Module):
    def __init__(self,
                 in_channels, out_channels, wn,
                 group=1):
        super(CGBs, self).__init__()

        self.CGB1 = CGBBlock(in_channels)
        self.CGB2 = CGBBlock(in_channels)
        self.CGB3 = CGBBlock(in_channels)

        self.reduction1 = BasicConv2d(wn, in_channels * 2, out_channels, 1, 1, 0)
        self.reduction2 = BasicConv2d(wn, in_channels * 3, out_channels, 1, 1, 0)
        self.reduction3 = BasicConv2d(wn, in_channels * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        CGB1 = self.NAF1(o0)
        concat1 = torch.cat([c0, CGB1], dim=1)
        out1 = self.reduction1(concat1)

        CGB2 = self.NAF2(out1)
        concat2 = torch.cat([concat1, CGB2], dim=1)
        out2 = self.reduction2(concat2)

        CGB3 = self.NAF3(out2)
        concat3 = torch.cat([concat2, CGB3], dim=1)
        out3 = self.reduction3(concat3)

        return out3


class Network(nn.Module):

    def __init__(self, **kwargs):
        super(Network, self).__init__()

        wn = lambda x: torch.nn.utils.weight_norm(x)
        upscale = kwargs.get("upscale")
        scale = kwargs.get("scale")
        group = kwargs.get("group", 4)

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))

        self.CGG1 = CGBs(64, 64, wn=wn)
        self.CGG2 = CGBs(64, 64, wn=wn)
        self.CGG3 = CGBs(64, 64, wn=wn)

        self.reduction1 = BasicConv2d(wn, 64 * 2, 64, 1, 1, 0)
        self.reduction2 = BasicConv2d(wn, 64 * 3, 64, 1, 1, 0)
        self.reduction3 = BasicConv2d(wn, 64 * 4, 64, 1, 1, 0)

        self.reduction = BasicConv2d(wn, 64 * 3, 64, 1, 1, 0)

        self.Global_skip = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(64, 64, 1, 1, 0), nn.ReLU(inplace=True))

        self.upsample1 = UpsampleBlock(64, upscale=upscale, wn=wn, group=group)
        self.upsample2 = UpsampleBlock(64, upscale=scale, wn=wn, group=group)
        self.upsample3 = UpsampleBlock(64, upscale=1, wn=wn, group=group)

        self.exit1 = wn(nn.Conv2d(64, 3, 3, 1, 1))
        self.exit2 = wn(nn.Conv2d(64, 3, 3, 1, 1))
        self.exit3 = wn(nn.Conv2d(64, 3, 3, 1, 1))

        self.x_scale = Scale(1)
        self.res_scale = Scale(1)

    def forward(self, x, scale, upscale):
        x = self.sub_mean(x)
        skip = x

        x = self.entry_1(x)

        c0 = o0 = x

        CGG1 = self.CGG1(o0)
        concat1 = torch.cat([c0, CGG1], dim=1)
        out1 = self.reduction1(concat1)

        CGG2 = self.CGG2(out1)
        concat2 = torch.cat([concat1, CGG2], dim=1)
        out2 = self.reduction2(concat2)

        CGG3 = self.CGG3(out2)
        concat3 = torch.cat([concat2, CGG3], dim=1)
        out3 = self.reduction3(concat3)

        output = self.reduction(torch.cat((out1, out2, out3), 1))

        output = self.x_scale(output) + self.res_scale(self.Global_skip(x))

        output1 = self.upsample1(output, upscale=scale + 1)
        output2 = self.upsample2(output, upscale=scale)
        output3 = self.upsample(output, upscale=1 / 2)

        output1 = F.interpolate(output1, (x.size(-2) * scale, x.size(-1) * scale), mode='bicubic', align_corners=False)
        output2 = F.interpolate(output2, (x.size(-2) * scale, x.size(-1) * scale), mode='bicubic', align_corners=False)
        output3 = F.interpolate(output3, (x.size(-2) * scale, x.size(-1) * scale), mode='bicubic', align_corners=False)
        skip = F.interpolate(skip, (x.size(-2) * scale, x.size(-1) * scale), mode='bicubic', align_corners=False)

        output = self.exit1(output1) + self.exit2(output2) + self.exit3(output3) + skip

        output = self.add_mean(output)

        return output
