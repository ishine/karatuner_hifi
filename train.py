import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from dataset import AIShell
from models import KaraTuner, MultiPeriodDiscriminator, MultiScaleDiscriminator
from utils import feature_loss, discriminator_loss, generator_loss, stft_loss
from utils import get_writer, save_checkpoint, load_checkpoint, scan_checkpoint
from eval import evaluation

import os
import warnings
import argparse
import itertools
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='settings')
parser.add_argument('--epoch', default=200, type=int, help='input epoch')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='num workers')
parser.add_argument('--num_gpus', default=1, type=int, help='num gpus')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--lr_decay', default=0.999, type=float, help='lr decay')
parser.add_argument('--b1', default=0.8, type=float, help='adam b1')
parser.add_argument('--b2', default=0.99, type=float, help='adam b2')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--report_step', default=50, type=int, help='report step')
parser.add_argument('--eval_interval', default=1, type=int, help='eval interval')
parser.add_argument('--checkpoint_interval', default=1, type=int, help='checkpoint interval')
parser.add_argument('--version', default='hifi1', type=str, help='writer folder')
parser.add_argument('--checkpoints', default='saves', type=str, help='checkpoints folder')
parser.add_argument('--cp_g', default=None, type=str, help='generator checkpoint')
parser.add_argument('--cp_do', default=None, type=str, help='else checkpoint')

args = parser.parse_args()


def train(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    G = KaraTuner(pitch_channel=100, sp_channel=1225, mid_channel=512, frames=430,
                  upsample_initial_channel=512)
    MPD = MultiPeriodDiscriminator()
    MSD = MultiScaleDiscriminator()

    os.makedirs(args.checkpoints, exist_ok=True)
    print("checkpoints directory : ", args.checkpoints)

    steps = 0
    evaluations = []

    # load checkpoint
    if args.cp_g is None or args.cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(args.checkpoints + '/' + args.cp_g, device)
        state_dict_do = load_checkpoint(args.checkpoints + '/' + args.cp_do, device)
        G.load_state_dict(state_dict_g['generator'])
        MPD.load_state_dict(state_dict_do['mpd'])
        MSD.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if args.num_gpus > 1:
        G = DataParallel(G, device_ids=args.ids)
        MPD = DataParallel(MPD, device_ids=args.ids)
        MSD = DataParallel(MSD, device_ids=args.ids)

    G = G.to(device)
    MPD = MPD.to(device)
    MSD = MSD.to(device)

    # optim
    optim_g = torch.optim.AdamW([{'params': G.parameters(), 'initial_lr': args.lr}],
                                args.lr, betas=[args.b1, args.b2], weight_decay=args.weight_decay)
    optim_d = torch.optim.AdamW([{'params': itertools.chain(MSD.parameters(), MPD.parameters()),
                                  'initial_lr': args.lr}],
                                args.lr, betas=[args.b1, args.b2], weight_decay=args.weight_decay)

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=args.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=args.lr_decay, last_epoch=last_epoch)

    # stft loss
    sp_loss = stft_loss()

    # clean ram
    if args.cp_g is None or args.cp_do is None:
        state_dict_g = {}
        state_dict_do = {}

    # dataset
    train_set = AIShell(folder='data/', table='data/test.csv', subset='train', frames=430)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_set = AIShell(folder='data/', table='data/test.csv', subset='train', frames=430)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=False)
    save_set = AIShell(folder='data/', table='data/test.csv', subset='train', frames=430)
    save_loader = DataLoader(save_set, batch_size=1, shuffle=False, pin_memory=False)
    writer = get_writer('log/', args.version)

    G.train()
    MPD.train()
    MSD.train()

    epochs = args.epoch

    for epoch in range(max(0, last_epoch), epochs):

        for sp, f0, y_r in tqdm(train_loader, total=len(train_loader)):

            # to device
            sp = sp.to(device)
            f0 = f0.to(device)
            y_r = y_r.to(device)

            # Discriminator
            optim_d.zero_grad()
            y_g = G(sp, f0)

            # MPD
            y_r_mpd, y_g_mpd, _, _ = MPD(y_r, y_g.detach())
            loss_mpd = discriminator_loss(y_r_mpd, y_g_mpd)

            # MSD
            y_r_msd, y_g_msd, _, _ = MSD(y_r, y_g.detach())
            loss_msd = discriminator_loss(y_r_msd, y_g_msd)

            loss_d = loss_mpd + loss_msd
            loss_d.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            loss_sp = sp_loss(y_r, y_g)*45

            y_r_mpd, y_g_mpd, fm_r_mpd, fm_g_mpd = MPD(y_r, y_g)
            y_r_msd, y_g_msd, fm_r_msd, fm_g_msd = MSD(y_r, y_g)

            loss_fm = feature_loss(fm_r_mpd, fm_g_mpd) + feature_loss(fm_r_msd, fm_g_msd)
            loss_gen = generator_loss(y_g_mpd) + generator_loss(y_g_msd)

            loss_g = loss_fm + loss_gen + loss_sp

            loss_g.backward()
            optim_g.step()

            # report
            if (steps+1) % args.report_step == 0:
                print('Epoch: [{}][{}]    Step: [{}]    D_Loss: {:.6f}  STFT_Loss: {:.6f}   G_Loss: {:.6f}'.format(
                    epoch + 1, epochs, steps + 1,  loss_d, loss_sp, loss_g))
                writer.add_scalar('D Loss', loss_d, global_step=steps)
                writer.add_scalar('G Loss', loss_g, global_step=steps)
                writer.add_scalar('Feature Loss', loss_fm, global_step=steps)
                writer.add_scalar('Generator Loss', loss_gen, global_step=steps)
                writer.add_scalar('STFT Loss', loss_sp, global_step=steps)

            steps += 1

        # validation
        if (epoch+1) % args.eval_interval == 0:
            evaluations.append(evaluation(G, val_loader, device, epoch+1, sr=22050, folder=False))
            evaluation(G, save_loader, device, epoch+1, sr=22050, folder='val/')

        # checkpoint
        if (epoch+1) % args.checkpoint_interval == 0:
            checkpoint_path = "{}/{}_g_{:04d}".format(args.checkpoints, args.version, epoch)
            save_checkpoint(checkpoint_path,
                            {'generator': (G.module if args.num_gpus > 1 else G).state_dict()})

            checkpoint_path = "{}/{}_do_{:04d}".format(args.checkpoints, args.version, epoch)
            save_checkpoint(checkpoint_path,
                            {'mpd': (MPD.module if args.num_gpus > 1
                                     else MPD).state_dict(),
                             'msd': (MSD.module if args.num_gpus > 1
                                     else MSD).state_dict(),
                             'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                             'epoch': epoch})

        scheduler_g.step()
        scheduler_d.step()


if __name__ == "__main__":
    if torch.cuda.is_available():
        args.num_gpus = torch.cuda.device_count()
        args.batch_size = int(args.batch_size / args.num_gpus)
        print('Batch size per GPU :', args.batch_size)
    else:
        pass
    args.ids = [i for i in range(args.num_gpus)]

    train(args)

