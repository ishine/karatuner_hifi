import glob
import os
import matplotlib

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

matplotlib.use("Agg")
import matplotlib.pylab as plt
import shutil
from tqdm import tqdm
from tensorboardX import SummaryWriter


def get_writer(folder, name):
    try:
        shutil.rmtree(folder+name)
        os.makedirs(folder+name)
    except:
        os.makedirs(folder+name)
    writer = SummaryWriter(folder+name)
    return writer


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]