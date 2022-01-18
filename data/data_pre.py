from multiprocessing import Process
import numpy as np
import pandas as pd
import pyworld as pw
import librosa
from tqdm import tqdm


def chunks(arr, m):
    result = [[] for i in range(m)]
    for i in range(len(arr)):
        result[i%m].append(arr[i])
    return result


def get_world(df, folder, fft_size, hop_length, sr, time):
    for i in tqdm(range(len(df))):
        line = df.iloc[i]
        path = folder+line['path']
        y, fs = librosa.load(path+'speech.wav', dtype=np.float64)
        if fs != sr:
            y = librosa.resample(y, fs, sr)

        # length limit
        if len(y) < time:
            y = np.append(y, np.zeros(time-len(y), dtype=np.float64), axis=-1)
        if len(y) > time:
            y = y[:time]

        f0, sp, ap = pw.wav2world(y[:time-1], sr, fft_size=fft_size, frame_period=hop_length/sr*1000)
        f0 = f0[np.newaxis, :]
        sp = sp.T

        np.save(path+'f0.npy', f0)
        np.save(path+'sp.npy', sp)


if __name__ == '__main__':
    cores = 2
    folder = 'data/'
    df = pd.read_csv('data/test.csv')
    fft_size = 2048
    hop_length = 512
    sr = 22050
    time = 220160

    lines = [i for i in range(len(df))]
    subsets = chunks(lines, cores)

    for subset in subsets:
        t = Process(target=get_world,
                    args=(df.iloc[subset], folder, fft_size, hop_length, sr, time,))
        t.start()




