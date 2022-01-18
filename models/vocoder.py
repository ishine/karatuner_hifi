import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from .utils import init_weights, get_padding
from .blocks import *
from .SF import *

LRELU_SLOPE = 0.1


class Generator(nn.Module):
    def __init__(self, in_channel=80, upsample_initial_channel=512, upsample_rates=None, upsample_kernel_sizes=None,
                 resblock='1', resblock_kernel_sizes=None, resblock_dilation_sizes=None):
        super(Generator, self).__init__()

        # default settings
        if upsample_rates is None:
            upsample_rates = [8, 8, 4, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 16, 8, 4]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        resblock = ResBlock1 if resblock == '1' else ResBlock2

        # In Conv
        self.conv_pre = weight_norm(Conv1d(in_channel, upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # UpSampling
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))
            # ResBlocks
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        # Out Conv
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        # init weights
        self.conv_pre.apply(init_weights)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class KaraTuner(nn.Module):
    def __init__(self, pitch_channel=1, sp_channel=1250, mid_channel=512, frames=430,
                 upsample_initial_channel=512, upsample_rates=None, upsample_kernel_sizes=None,
                 resblock='1', resblock_kernel_sizes=None, resblock_dilation_sizes=None):
        super(KaraTuner, self).__init__()
        self.SF = SourceFilter(pitch_channel, sp_channel, mid_channel, frames=frames)
        self.Vocoder = Generator(mid_channel,  upsample_initial_channel, upsample_rates, upsample_kernel_sizes,
                           resblock, resblock_kernel_sizes, resblock_dilation_sizes)

    def forward(self, sp, pitch):
        x = self.SF(sp, pitch)
        x = self.Vocoder(x)
        return x

    def remove_weight_norm(self):
        self.SF.remove_weight_norm()
        self.G.remove_weight_norm()

'''
G = KaraTuner()
pitch = torch.rand(4, 1, 430)
sp = torch.rand(4, 1250, 430)
test = G(sp, pitch)
'''