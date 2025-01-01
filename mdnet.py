import os

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from models.sr3_modules.se import ChannelSpatialSELayer
from models.fusion import AWFM




def create_wavelet_filter(wave, in_size, cuda):
    w = pywt.Wavelet(wave)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float, device='cuda')
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float, device='cuda')

    filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)

    filters = filters[:, None].repeat(in_size, 1, 1, 1)
    return filters

class WTconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, wavelet_type='db1', levels=3):
        super(WTconv2d, self).__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.levels = levels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确定设备
        self.wt_filter = create_wavelet_filter(wavelet_type, in_channels, self.device)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, padding=1, groups=in_channels)
            for _ in range(levels)
        ])
        self.conv1 = nn.Conv2d(in_channels * 3,out_channels,kernel_size=3, padding=1)
    def forward(self, x):
        x = x.to(self.device)
        outputs = []

        for level in range(self.levels):
            x_wavelet = F.conv2d(x, self.wt_filter, stride=1, padding='same', groups=self.in_channels)
            x_wavelet = x_wavelet.view(x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3])

            x_wavelet = self.convs[level](x_wavelet)
            outputs.append(x_wavelet)
            x = x_wavelet
        y = torch.cat(outputs, dim=1)
        y = self.conv1(y)
        return y




def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    '''
    Get the number of input layers to the change detection head.
    '''
    in_channels = 0
    for scale in feat_scales:
        if scale < 3:  # 256 x 256
            in_channels += inner_channel * channel_multiplier[0]
        elif scale < 6:  # 128 x 128
            in_channels += inner_channel * channel_multiplier[1]
        elif scale < 9:  # 64 x 64
            in_channels += inner_channel * channel_multiplier[2]
        elif scale < 12:  # 32 x 32
            in_channels += inner_channel * channel_multiplier[3]
        elif scale < 15:  # 16 x 16
            in_channels += inner_channel * channel_multiplier[4]
        else:
            print('Unbounded number for feat_scales. 0<=feat_scales<=14')
    return in_channels


class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
        )

    def forward(self, x):
        return self.block(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim * len(time_steps), dim, 1)
            if len(time_steps) > 1
            else None,
            nn.ReLU()
            if len(time_steps) > 1
            else None,
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class HeadTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        # return torch.tanh(self.conv(x))   # (-1, 1)
        return torch.tanh(self.conv(x))   # (-1, 1)



class HeadLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.4)

# class HeadLeakyRelu2d_none(nn.Module):
#     # convolution
#     # leaky relu
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
#         super(HeadLeakyRelu2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
#
#     def forward(self, x):
#         return F.leaky_relu(self.conv(x), negative_slope=0.4)

class BBasicconv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BBasicconv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

class Fusion_module(nn.Module):
    '''
    基于注意力的自适应特征聚合 Fusion_Module
    '''

    def __init__(self, channels=64, r=4):
        super(Fusion_module, self).__init__()
        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, 2 * inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(2 * inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * inter_channels, 2 * channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(2 * channels),
            nn.Sigmoid(),
        )

        self.channel_agg = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            )

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input ## 先对特征进行一步自校正
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim =1)
        agg_input = self.channel_agg(recal_input) ## 进行特征压缩 因为只计算一个特征的权重
        local_w = self.local_att(agg_input)  ## 局部注意力 即spatial attention
        global_w = self.global_att(agg_input) ## 全局注意力 即channel attention
        w = self.sigmoid(local_w * global_w) ## 计算特征x1的权重
        xo = w * x1 + (1 - w) * x2 ## fusion results ## 特征聚合
        return xo

class DSFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(DSFM, self).__init__()
        self.RGBobj1_1 = BBasicconv2d(in_C, out_C, 3, 1, 1)
        self.RGBobj1_2 = BBasicconv2d(out_C, out_C, 3, 1, 1)
        self.RGBspr = BBasicconv2d(out_C, out_C, 3, 1, 1)

        self.Infobj1_1 = BBasicconv2d(in_C, out_C, 3, 1, 1)
        self.Infobj1_2 = BBasicconv2d(out_C, out_C, 3, 1, 1)
        self.Infspr = BBasicconv2d(out_C, out_C, 3, 1, 1)
        self.obj_fuse = Fusion_module(channels=out_C)

    def forward(self, rgb, depth):
        rgb_sum = self.RGBobj1_2(self.RGBobj1_1(rgb))
        rgb_obj = self.RGBspr(rgb_sum)
        Inf_sum = self.Infobj1_2(self.Infobj1_1(depth))
        Inf_obj = self.Infspr(Inf_sum)
        out = self.obj_fuse(rgb_obj, Inf_obj)
        return out

class BasicUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3):
        super(BasicUnit, self).__init__()
        p = kernel_size // 2
        self.basic_unit = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=p, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=p, bias=False)
        )

    def forward(self, input):
        return self.basic_unit(input)





class selfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(selfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = 1.0 / (out_channels ** 0.5)

    def forward(self, feature, feature_map):
        query = self.query_conv(feature)
        key = self.key_conv(feature)
        value = self.value_conv(feature)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores * self.scale

        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_weights, value)

        output_feature_map = (feature_map + attended_values)

        return output_feature_map

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.conv_ch = ChannelAttention(channel)
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        x = x * y
        # y1 = self.conv_ch(x)
        return x

class mixblock(nn.Module):
    def __init__(self, n_feats):
        super(mixblock, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())
        self.conv2=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())
        self.alpha=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.ones(1))
    def forward(self,x):
        return self.alpha*self.conv1(x)+self.beta*self.conv2(x)

def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)

class frBlock(nn.Module):
    def __init__(self,
                 ms_channels,
                 n_feat):
        super(frBlock, self).__init__()

        self.down = nn.AvgPool2d(kernel_size=2)
        self.ega = selfAttention(n_feat, n_feat)
        self.raw_alpha = nn.Parameter(torch.ones(1))
        # fill 0
        self.decoder_high = mixblock(n_feat)  # nn.Sequential(one_module(n_feats),
        self.decoder_low = nn.Sequential(mixblock(n_feat), mixblock(n_feat), mixblock(n_feat))
        self.raw_alpha.data.fill_(0)
        self.conv1 = nn.Conv2d(n_feat*2,n_feat,3,1,1,bias=False)
        self.conv2 = nn.Conv2d(n_feat,n_feat,1,1,0,bias=False)

        self.calayer = CALayer(n_feat)
        self.get_edge = WTconv2d(n_feat, n_feat)

    def forward(self,HR,LR,iteration):
        h1 = self.down(LR)
        high1 = LR - F.interpolate(h1, size=LR.size()[-2:], mode='bilinear', align_corners=True)
        h2 = self.down(HR)
        high2 = HR - F.interpolate(h2, size=LR.size()[-2:], mode='bilinear', align_corners=True)
        high1=high1+self.ega(high1,high1)*self.raw_alpha
        high2=high2+self.ega(high2,high2)*self.raw_alpha
        x1=self.decoder_low(h1)
        x2=self.decoder_low(h2)
        x3 = F.interpolate(x1, size=LR.size()[-2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x2, size=HR.size()[-2:], mode='bilinear', align_corners=True)
        high1 = self.decoder_high(high1)
        high2 = self.decoder_high(high2)
        high = high1 - high2
        low = x3 - x4
        x = self.calayer(self.conv1(torch.cat([low, high], dim=1))) + HR + LR
        y = self.get_edge(x)
        y = y + LR

        return y



class DepthwiseSeparableconv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableconv2d, self).__init__()
# Depthwise Convolution
        self.depthwise = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                           groups=in_channels, bias=True, padding_mode='zeros')
# Pointwise Convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1,
                           bias=True, padding_mode='zeros')


    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x




class Fusion_Head(nn.Module):
    '''
    Change detection head (version 2).
    '''

    def __init__(self, feat_scales, out_channels=3, inner_channel=None, channel_multiplier=None, img_size=256,
                 time_steps=None):
        super(Fusion_Head, self).__init__()

        # Define the parameters of the change detection head
        feat_scales.sort(reverse=True)
        self.feat_scales = feat_scales
        self.in_channels = get_in_channels(feat_scales, inner_channel, channel_multiplier)
        self.img_size = img_size
        self.time_steps = time_steps

        # Convolutional layers before parsing to difference head
        self.decoder = nn.ModuleList()
        for i in range(0, len(self.feat_scales)):
            dim = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)

            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=time_steps)
            )

            if i != len(self.feat_scales) - 1:
                dim_out = get_in_channels([self.feat_scales[i + 1]], inner_channel, channel_multiplier)
                self.decoder.append(
                    AttentionBlock(dim=dim, dim_out=dim_out)
                )
        # Final head
        self.rgb_decode2 = HeadLeakyRelu2d(128, 64)
        self.rgb_decode2_1 = HeadLeakyRelu2d(64, 64)
        self.rgb_decode1 = HeadTanh2d(64, 32)
        self.DSFM = DSFM(in_C=64, out_C=64)
        self.ms_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pan_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
        self.con1_1 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.DSconv = DepthwiseSeparableconv2d(64,64)
        self.fr_blocks = nn.ModuleList([frBlock(64, 64) for i in range(9)])
        self.cgafusion = AWFM(64, 8)

    def forward(self, feats, data):

        image_vis = data["vis"]
        image_ir = data["ir"]
        image_ir = self.pan_conv(image_ir)
        image_vis = self.ms_conv(image_vis)
        a = self.DSFM(image_vis, image_ir)
        for i in range(len(self.fr_blocks)):
            lvl = 0
            tar1 = self.fr_blocks[i](a, image_vis)  # tar1 2 64 256 256
            tar2 = self.fr_blocks[i](a, image_ir)  # 网络中A-BLOCK，tar11 2 64 256 256
            ir_mask =  self.MEEM(image_ir)
            tar1 = self.conv2(torch.cat((ir_mask,tar1),dim=1))
            tar2 = self.conv2(torch.cat((ir_mask,tar2),dim=1))

            a = self.cgafusion(tar1, tar2)
            a = self.rc_conv(a)


            for layer in self.decoder:
                if isinstance(layer, Block):
                    f_s = feats[0][self.feat_scales[lvl]]  # feature stacked
                    if len(self.time_steps) > 1:
                        for i in range(1, len(self.time_steps)):
                            f_s = torch.cat((f_s, feats[i][self.feat_scales[lvl]]), dim=1)
                    f_s = layer(f_s)
                    if lvl != 0:
                        f_s = f_s + x
                    lvl += 1
                else:
                    f_s = layer(f_s)
                    x = F.interpolate(f_s, scale_factor=2, mode="bilinear", align_corners=True)

            rgb_img = self.rc_conv(x = self.rgb_decode2(x))
            a = torch.cat((a,rgb_img), dim=1)
            a = self.DSconv(a)

        a = torch.cat((a, image_ir), dim=1)
        rgb_img = self.conv1(a)
        rgb_img = self.con1_1(rgb_img)
        return rgb_img