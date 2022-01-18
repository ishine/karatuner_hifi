import torch.nn as nn
import torch.nn.functional as F
import torch


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    # r_losses = []
    # g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        # r_losses.append(r_loss.item())
        # g_losses.append(g_loss.item())

    return loss


def generator_loss(disc_outputs):
    loss = 0
    # gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        # gen_losses.append(l)
        loss += l

    return loss


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x = torch.nn.functional.pad(x.unsqueeze(1), (int((fft_size-hop_size)/2),
                                                 int((fft_size-hop_size)/2)), mode='reflect').squeeze(1)

    x_stft = torch.stft(x, fft_size, hop_size, win_length, window,
                        center=False, pad_mode='reflect', normalized=False, onesided=True)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7))


class stft_loss(nn.Module):
    """STFT loss module."""
    def __init__(self, fft_size=2048, hop_size=512, win_length=2048, window="hann_window"):
        """Initialize STFT loss module."""
        super(stft_loss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)

    def forward(self, y_r, y_g):
        """Calculate forward propagation.
        Args:
            y (Tensor): Predicted signal (B, 1, T).
        Returns:
        """
        y_r = y_r.squeeze(1)
        y_g = y_g.squeeze(1)

        y_r_mag = stft(y_r, self.fft_size, self.hop_size, self.win_length, self.window.to(y_r.device))
        y_g_mag = stft(y_g, self.fft_size, self.hop_size, self.win_length, self.window.to(y_g.device))

        return F.l1_loss(y_r_mag, y_g_mag)


