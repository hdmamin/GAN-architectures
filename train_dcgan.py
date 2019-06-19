import argparse
import os

import matplotlib
import subprocess
# Must set matplotlib display before importing plotting functions from utils.
matplotlib.use('Agg')


from config import device
from models import (Generator, Discriminator, PretrainedDiscriminator, GRelu,
                    JRelu)
from torch_datasets import *
from training import train
from utils import render_samples, plot_output


def get_args():
    """Create argument parser to read settings from command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                        help='options: photo, sketch, quickdraw, celeb, small')
    parser.add_argument('-e', '--epochs', type=int,
                        help='# of epochs to train for.')
    parser.add_argument('-n', '--norm', type=str, default='bn',
                        help='bn for batch norm, in for instance norm')
    parser.add_argument('-l', '--leak', type=float, default=.2,
                        help='parameter for leaky ReLU')
    parser.add_argument('-j', '--jrelu', type=bool, default=False,
                        help='if True, use JRelu() as activation')
    parser.add_argument('-p', '--pretrained', type=bool, default=False,
                        help='Whether to use a pre-trained discriminator.')
    parser.add_argument('-s', '--sample_dir', type=str, default=None,
                        help='location to save samples')
    parser.add_argument('-q', '--quiet_mode', type=bool, default=True,
                        help='Limit printed output from training loop.')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate')
    parser.add_argument('--lr2', type=float, default=None,
                        help='D learning rate (if different than G lr).')
    parser.add_argument('--gd_ratio', type=int, default=1,
                        help='To train G more frequently than D, choose > 1.')
    parser.add_argument('--d_head_start', type=int, default=0,
                        help='# of mini batches to train D before training G.')
    parser.add_argument('--sample_freq', type=int, default=5,
                        help='When to generate samples (measured in epochs).')
    parser.add_argument('--stop_vm', type=bool, default=False,
                        help='Whether to close the VM when finished training.')
    args = parser.parse_args()

    return args


def main(args):
    """Train DCGAN for user-specified number of epochs."""
    dataloaders = dict(photo=photo_dl,
                       sketch=sketch_dl,
                       quickdraw=quickdraw_dl,
                       celeb=celeb_dl,
                       small=small_dl)
    dloader = dataloaders[args.dataset]
    if args.jrelu:
        activation = JRelu
    else:
        activation = GRelu(args.leak)
    
    # Set learning rates.
    if args.lr and args.lr2:
        lr = [args.lr, args.lr2]
    else:
        lr = args.lr
        
    # Initialize models.
    d = Discriminator(act=activation, norm=args.norm).to(device)
    if args.pretrained:
        g = PretrainedDiscriminator(activation).to(device)
    else:
        g = Generator(act=activation, norm=args.norm).to(device)
    output = train(args.epochs, dloader, lr, sample_freq=args.sample_freq,
                   sample_dir=args.sample_dir, d_head_start=args.d_head_start,
                   gd_ratio=args.gd_ratio, d=d, g=g)
    
    # Save loss plots, sample gif, and settings in sample_dir.
    if args.sample_dir:
        plot_output(output, args.sample_dir)
        render_samples(args.sample_dir)
        with open(os.path.join(args.sample_dir, 'README.md'), 'w') as f:
            f.write(str(args))

    # Close Google Cloud VM when done.
    if args.stop_vm:
        subprocess.run('gcloud compute instances stop my-fastai-instance '
                       '--zone=us-west2-b')


if __name__ == '__main__':
    args = get_args()
    main(args)
