import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

from utils import stft_loss

import soundfile as sf
from tqdm import tqdm


def evaluation(G, val_loader, device, epoch, sr, folder=False):
    G.eval()
    val_loss = 0.0
    step = 0
    sp_loss = stft_loss()
    with torch.no_grad():
        for sp, f0, y_r in tqdm(val_loader, total=len(val_loader)):
            # to device
            sp = sp.to(device)
            f0 = f0.to(device)
            y_r = y_r.to(device)

            y_g = G(sp, f0)

            val_loss += sp_loss(y_r, y_g)

            if folder:
                y_g = y_g.squeeze(0)
                y_g = y_g.squeeze(0)
                y_g = y_g.to('cpu').numpy()
                sf.write(folder+str(epoch)+'_'+str(step)+'.wav', y_g, sr)

            step += 1

        print('Val Loss: {:.6f}'.format(val_loss / step))

    G.train()
    return val_loss / step