import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from .utils import init_weights, get_padding
from .blocks import *

LRELU_SLOPE = 0.1


class SourceFilter(nn.Module):
    def __init__(self, pitch_channel, sp_channel, mid_channel,
                 res_hidden=None, dilation=None, kernel_size=3,
                 frames=1000):
        super(SourceFilter, self).__init__()

        if dilation is None:
            dilation = [1, 2, 1, 2]
        if res_hidden is None:
            res_hidden = mid_channel

        # Pitch
        self.pitch_embedding = weight_norm(Conv1d(pitch_channel, mid_channel, 1, 1, padding=0))
        # Sp
        self.sp_embedding = weight_norm(Conv1d(sp_channel, mid_channel, 1, 1, padding=0))

        # res blocks
        self.res_sp = ResBlockSF(mid_channel, res_hidden, kernel_size, dilation)
        self.res_ap = ResBlockSF(mid_channel, res_hidden, kernel_size, dilation)

        # learnable ratio
        self.ratio = nn.Parameter(torch.ones(frames), requires_grad=True)

        # out fc
        self.conv_post = weight_norm(Conv1d(mid_channel, mid_channel, 1, 1, padding=0))

        # init weights
        self.pitch_embedding.apply(init_weights)
        self.sp_embedding.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, sp, pitch):
        # embedding
        pitch = F.leaky_relu(self.pitch_embedding(pitch), LRELU_SLOPE)
        sp = F.leaky_relu(self.sp_embedding(sp), LRELU_SLOPE)

        # res blocks
        f_sp = F.leaky_relu(self.res_sp(sp), LRELU_SLOPE)
        ap = F.leaky_relu(self.res_ap(sp), LRELU_SLOPE)

        # source filter
        f_sp = self.ratio * f_sp
        f_sp = f_sp * pitch * sp
        output = f_sp + ap

        # output
        output = F.leaky_relu(output, LRELU_SLOPE)
        output = self.conv_post(output)

        return output

    def remove_weight_norm(self):
        print('Removing weight norm...')
        self.res_sp.remove_weight_norm()
        self.res_ap.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)