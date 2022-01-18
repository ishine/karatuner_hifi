import pandas as pd
import numpy as np
import librosa
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


def f0_to_2d(f0, sr, n_fft, f0_max):
    # default shape = 100
    f0 = f0[0]
    shape0 = max(int(f0_max*n_fft/sr+1), 100)
    f0_2d = np.zeros((shape0, len(f0)))
    idx1 = (f0*n_fft/sr).astype(int)
    idx2 = np.array(range(f0_2d.shape[1]))
    f0_2d[idx1, idx2] = 1
    f0_2d = np.array(f0_2d, dtype=np.float32)
    return f0_2d


class AIShell(Dataset):
    def __init__(
            self,
            folder,
            table,
            subset='train',
            frames=430,
            hop_length=512,
            shifting=200,
            f0_type='2d',
            augment=True
    ):
        self.folder = folder
        df = pd.read_csv(table)
        df = df[df['subset'] == subset]
        self.df = df
        self.frames = frames
        self.shifting = shifting
        self.hop_length = hop_length
        self.f0_type = f0_type
        self.augment = augment

    def __getitem__(self, i):
        path = self.df.iloc[i]['path']

        sp = np.load(self.folder+path+'sp.npy')
        f0 = np.load(self.folder+path+'f0.npy')
        audio, fs = librosa.load(self.folder+path+'speech.wav', dtype=np.float32)
        audio = audio[np.newaxis, :]

        if self.frames:
            if sp.shape[-1] < self.frames:
                sp = np.append(sp, np.zeros((sp.shape[0], self.frames-sp.shape[1]),
                                            dtype=np.float32), axis=-1)
                f0 = np.append(f0, np.zeros((f0.shape[0], self.frames-f0.shape[1]),
                                            dtype=np.float32), axis=-1)

        if audio.shape[1] < self.frames*self.hop_length:
            audio = np.append(audio, np.zeros((audio.shape[0],
                                               self.frames*self.hop_length-audio.shape[1]),
                                              dtype=np.float32), axis=-1)

        if self.shifting:
            new_sp = np.zeros((sp.shape[0]+self.shifting, sp.shape[1]), dtype=np.float32)
            if self.augment:
                shift_num = random.randint(0, self.shifting-1)
            else:
                shift_num = 0
            new_sp[shift_num:sp.shape[0]+shift_num, :] = sp
            sp = new_sp

        # output
        sp = np.array(sp, dtype=np.float32)[:, :self.frames]
        f0 = np.array(f0, dtype=np.float32)[:, :self.frames]
        audio = np.array(audio, dtype=np.float32)[:, :self.frames*self.hop_length]

        if self.f0_type == '2d':
            f0 = f0_to_2d(f0, 22050, 2048, 1000)

        return sp, f0, audio

    def __len__(self):
        return len(self.df)

