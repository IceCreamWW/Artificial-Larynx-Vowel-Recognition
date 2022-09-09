import argparse
import numpy as np
from espnet2.asr.frontend.default import DefaultFrontend
from tqdm import tqdm
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def norm_wav(wav):
    norm = max(np.absolute(wav))
    if norm > 1e-5:
        wav = wav / norm
    return wav

def make_image(csv_path, img_path, args):
    wav = torch.FloatTensor(np.genfromtxt(csv_path))
    wav = norm_wav(wav)

    wav.unsqueeze_(0)
    wav_length = torch.LongTensor([wav.shape[1]])
    frontend = DefaultFrontend(fs=args.fs, n_fft=args.n_fft, win_length=args.win_length, hop_length=args.hop_length, fmin=args.fmin, fmax=args.fmax, n_mels=args.n_mels)
    feats, feats_length = frontend(wav, wav_length)
    feats = feats.squeeze(0).transpose(0, 1).flip(0)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(args.size / args.dpi, args.size / args.dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.matshow(feats, cmap=args.cmap)
    plt.savefig(img_path, dpi=args.dpi)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make jpgs from csv file list")
    parser.add_argument("-f", "--flist", type=str, required=True)
    parser.add_argument("--fs", type=int, default=48000)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=8000)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--win-length", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--cmap", type=str, default="jet", help="jet, gray_r, etc.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=32)
    args = parser.parse_args()

    with open(args.flist) as fp:
        csv_paths = [csv_path.strip() for csv_path in fp.readlines() if len(csv_path.strip()) != 0]

    for csv_path in tqdm(csv_paths):
        img_path = csv_path.removesuffix("csv") + "jpg"
        make_image(csv_path, img_path, args)

