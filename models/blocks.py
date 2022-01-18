import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from .utils import init_weights, get_padding

LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])

        for i in range(len(self.convs1)):
            self.convs1[i].apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])

        for i in range(len(self.convs2)):
            self.convs2[i].apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])

        for i in range(len(self.convs)):
            self.convs[i].apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class ResBlockSF(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, dilation=None):
        if dilation is None:
            dilation = [1, 2, 1, 2]
        super(ResBlockSF, self).__init__()
        self.conv_pre = weight_norm(Conv1d(in_channels, hidden_channels, 1, 1, dilation=1,
                                           padding=0))
        convs = []
        for d in dilation:
            convs.append(weight_norm(Conv1d(hidden_channels, hidden_channels, kernel_size, 1, dilation=d,
                                            padding=get_padding(kernel_size, d))))
        self.convs = nn.ModuleList(convs)

        self.conv_post = weight_norm(Conv1d(hidden_channels, in_channels, 1, 1, dilation=1,
                                            padding=0))

        self.conv_pre.apply(init_weights)
        self.conv_post.apply(init_weights)
        for i in range(len(self.convs)):
            self.convs[i].apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


