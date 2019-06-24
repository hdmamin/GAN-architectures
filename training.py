from collections import defaultdict
from collections.abc import Iterable
from itertools import chain
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from models import Discriminator, Generator, CycleGenerator
from config import *


def get_optimizers(g, d, lr_g, lr_d, b1=.5):
    """Return Adam optimizers for generator and discriminator.
    
    Parameters
    -----------
    g: Generator
    d: Discriminator
    lr_g: float
        Learning rate for generator.
    lr_d: float
        Learning rate for discriminator.
    b1: float
        Hyperparameter for Adam.
    """
    g_optim = torch.optim.Adam(g.parameters(), lr=lr_g, betas=(b1, .999))
    d_optim = torch.optim.Adam(d.parameters(), lr=lr_d, betas=(b1, .999))
    return g_optim, d_optim

                      
def train(epochs, dl, lr=2e-4, b1=.5, sample_freq=5, sample_dir='samples',
          save_weights=False, d_head_start=0, gd_ratio=1, quiet_mode=True, 
          g=None, d=None):
    """Train generator and discriminator with Adam.

    Parameters
    -----------
    epochs: int
        Number of epochs to train for.
    dl: DataLoader
    lr: float
        Learning rate. Paper recommends .0002.
    b1: float
        Hyperparameter for Adam. Paper recommends 0.5.
    sample_freq: int
        Save sample images from generator every n epochs (default 10).
    sample_dir: str
        Directory to save images output by the generator.
    save_weights: bool
        If True, save weights in the sample directory with the same 
        frequency as samples are generated. Default is False,
        meaning no weights will be saved.
    d_head_start: int
        Number of batches to train D for before starting to train G. Default of
        zero means both will be trained from the beginning. The hope is that
        a nonzero value can be used to make D a more useful teacher early in
        training.
    gd_ratio: int
        Number of times to train G for each time we train D. In other words,
        G will be trained every mini batch, but D will only be trained every
        gd_ratio epochs. If D loss is very low and G loss is high, the hope is
        that increasing gd_ratio will allow G to learn more and "catch up" to
        D.
    quiet_mode: bool
        If True, limit printed epoch output to a max of 20 times (e.g. for 100
        epochs, it will print results at every fifth epoch). Printing more
        frequently can incapacitate the notebook when training for many epochs.
    d: Discriminator, subclass of nn.Module
        Discriminator, a binary classifier (1 for real, 0 for fake).
    g: Generator, subclass of nn.Module
        Generator that upsamples random noise into image.
    """
    if not g:
        g = Generator().to(device)
    if not d:
        d = Discriminator().to(device)
    # For GANs, models should stay in train mode.
    g.train()
    d.train()

    # Define loss and optimizers.
    if isinstance(lr, Iterable):
        lr1, lr2 = lr
    else:
        lr1 = lr2 = lr
    g_optim, d_optim = get_optimizers(g, d, lr1, lr2, b1)
    criterion = nn.BCELoss()                
    
    # Noise used for sample images, not training.
    fixed_noise = torch.randn(dl.batch_size, 100, 1, 1, device=device)

    # Calculate printing frequency and sample frequency.
    print_freq = 1
    if quiet_mode:
        print_freq = max(1, int(np.ceil(epochs / 20)))
    iter_sample_freq = len(dl) * sample_freq

    # Make directories for samples and weights.
    os.makedirs(sample_dir, exist_ok=True)

    # Store stats to return at end.
    stats = defaultdict(list)
    iters = 0
    
    # Train D and G.
    for epoch in range(epochs):
        for i, (x, y) in enumerate(dl):
            x = x.to(device)
            bs_curr = x.shape[0]
            real_labels = torch.ones(bs_curr, device=device)
            fake_labels = torch.zeros(bs_curr, device=device)
            noise = torch.randn(bs_curr, 100, 1, 1, device=device)
            train_d = ((iters % gd_ratio == 0) or 
                       (epoch == 0 and iters < d_head_start))
            train_g = (epoch > 0 or i >= d_head_start)

            ##################################################################
            # Train discriminator. Detach G output for speed.
            ##################################################################
            d_optim.zero_grad()
            fake = g(noise)
            y_hat_fake = d(fake.detach())
            y_hat_real = d(x)

            if train_d:
                # Compute losses.
                d_loss_fake = criterion(y_hat_fake, fake_labels)
                d_loss_real = criterion(y_hat_real, real_labels)
                d_loss = d_loss_real + d_loss_fake

                # Update stats.
                fake_avg = y_hat_fake.mean().item()
                real_avg = y_hat_real.mean().item()
                stats['d_fake_avg'].append(fake_avg)
                stats['d_real_avg'].append(real_avg)
                stats['d_fake_loss'].append(d_loss_fake.item())
                stats['d_real_loss'].append(d_loss_real.item())

                # Backpropagation.
                d_loss.backward()
                d_optim.step()

            ##################################################################
            # Train generator. Use true_labels because we want to fool D.
            # G's weights are the same so no need to re-generate fake
            # examples. D was updated so compute loss again.
            ##################################################################
            if train_g:
                g_optim.zero_grad()
                g_loss = criterion(d(fake), real_labels)
                stats['g_loss'].append(g_loss.item())
                g_loss.backward()
                g_optim.step()

            # Generate sample from fixed noise at end of every nth iteration.
            if iters % iter_sample_freq == 0:
                with torch.no_grad():
                    fake = g(fixed_noise)
                    vutils.save_image(fake.detach(), f'{sample_dir}/{epoch+1}.png',
                                      normalize=True)

                # If specified, save weights corresponding to generated samples.
                if save_weights:
                    states = dict(g=g.state_dict(),
                                  d=d.state_dict(),
                                  g_optimizer=g_optim.state_dict(),
                                  d_optimizer=d_optim.state_dict(),
                                  epoch=epoch)
                    torch.save(states, f'{sample_dir}/{epoch+1}.pth')
            iters += 1
                      
        # Losses sometimes get stuck at zero - try increasing LR to escape.
        if (stats['d_fake_loss'][-1] == stats['d_fake_loss'][-2] and 
            stats['d_real_loss'][-1] == stats['d_real_loss'][-2]):
            lr1 *= 1.2
            lr2 *= 1.2
            g_optim, d_optim = get_optimizers(g, d, lr1, lr2, b1)
        
        # Print losses.
        if epoch % print_freq == 0:
            print(f'\nEpoch [{epoch+1}/{epochs}] \nBatch {i+1} Metrics:')
            if not train_d:
                print('>>>SKIP D TRAINING. Most recent stats:')
            print(f"D loss (real): {stats['d_real_loss'][-1]:.4f}\t"
                  f"D loss (fake): {stats['d_fake_loss'][-1]:.4f}")
            if train_g:
                print(f"G loss: {stats['g_loss'][-1]:.4f}")

    torch.save(stats, f'{sample_dir}/stats.pkl')
    return stats
                      

def get_cycle_models(path=None):
    """Get 2 cycle generators and 2 discriminators. If a path to a weight file
    is provided, load state dicts from that file. Models are returned in train
    mode.

    Parameters
    -----------
    path: str
        Optional - pass in path to weights file to load previously saved state
        dicts, or exclude to get new models.
    """
    models = dict(g_xy=CycleGenerator(), g_yx=CycleGenerator(),
                  d_x=Discriminator(), d_y=Discriminator())
    if path:
        states = torch.load(path)
        print(f"Loading models from epoch {states['epoch']}.")
    for name, model in models.items():
        if path:
            model.load_state_dict(states[name])
        model.to(device)
        model.train()
    print('All models currently in training mode.')
    return list(models.values())


def train_cycle_gan(epochs, x_dl, y_dl, sample_dir, save_weights=False, 
                    sample_freq=10, lr=2e-4, b1=.5, loss_type='bce', 
                    quiet_mode=True, load_path=None, models=None):
    """Train cycleGAN with Adam optimizer. The naming conventin G_xy will be
    used to refer to a generator that converts from set x to set y, while
    D_x refers to a discriminator that classifies examples as actually 
    belonging to set x (class 1) or being a model-generated example (class 0).
    
    Parameters
    -----------
    loss_type: bool
        Specifies whether to use class labels (i.e. horse, zebra, giraffe). If
        False, D only tries to predict if it is a real or fake example 
        (e.g. photo or sketch2photo).
    load_path: str or None
        If str, models will load state dicts from the provided file. If None,
        new models will be created.
    models: list or None
        Instead of providing a load path, we can also pass in a list of models
        in the form [G_xy, G_yx, D_x, D_y]. Either load_path or models 
        (or both) should be None.
    """
    # Create models. GANs should stay in train mode.
    if not models:
        G_xy, G_yx, D_x, D_y = get_cycle_models(load_path)
    else:
        for model in models:
            model.to(device)
            model.train()
        G_xy, G_yx, D_x, D_y = models
    
    # Create optimizers.
    if isinstance(lr, Iterable):
        lr1, lr2 = lr
    else:
        lr1 = lr2 = lr
    optim_g = torch.optim.Adam(chain(G_xy.parameters(), G_yx.parameters()), 
                               lr1, betas=(b1, .999))
    optim_d = torch.optim.Adam(chain(D_x.parameters(), D_y.parameters()),
                               lr2, betas=(b1, .999))    
    
    # Define loss function.
    if loss_type == 'bce':
        criterion = nn.BCELoss(reduction='mean')
    elif loss_type == 'mse':
        criterion = nn.MSELoss(reduction='mean')
    elif loss_type == 'mae':
        criterion = nn.L1Loss(reduction='mean')
        
    # Set fixed examples for sample generation.
    fixed_x = next(iter(x_dl))[0]
    fixed_y = next(iter(y_dl))[0]
    
    # Suppress printed output to 20 total times to avoid slowing down nb.
    print_freq = 1
    if quiet_mode:
        print_freq = max(1, epochs // 20)
    
    # Store lists of losses.
    stats = defaultdict(list)
    
    # Make dirs to store samples and weights.
    sample_dir_x = os.path.join(sample_dir, 'x')
    sample_dir_y = os.path.join(sample_dir, 'y')
    os.makedirs(sample_dir_x, exist_ok=True)
    os.makedirs(sample_dir_y, exist_ok=True)
    
    for epoch in range(epochs):
        for i, ((x, x_labels), (y, y_labels)) in enumerate(zip(x_dl, y_dl)):
            x = x.to(device)
            y = y.to(device)
            batch_len = x.shape[0]
            # X has less data so DL runs out and last batch is small. FIX?
            if batch_len != y.shape[0]:
                continue
            labels_real = torch.ones(batch_len, device=device)
            labels_fake = torch.zeros(batch_len, device=device)
            
            ##################################################################
            # Train D_x and D_y.
            ##################################################################
            # Train D's on real images.
            optim_d.zero_grad()
            pred_x, pred_y = D_x(x), D_y(y)
            loss_dx = criterion(pred_x, labels_real)
            loss_dy = criterion(pred_y, labels_real)
            loss_d_real = loss_dx + loss_dy
            stats['d_real'].append(loss_d_real.item())
            loss_d_real.backward()
            optim_d.step()
            
            # Train D's on fake images.
            optim_d.zero_grad()
            x_fake, y_fake = G_yx(y), G_xy(x)
            pred_x, pred_y = D_x(x_fake), D_y(y_fake)
            loss_dx_fake = criterion(pred_x, labels_fake)
            loss_dy_fake = criterion(pred_y, labels_fake)
            loss_d_fake = loss_dx_fake + loss_dy_fake
            stats['d_fake'].append(loss_d_fake.item())
            loss_d_fake.backward()
            optim_d.step()
            
            ##################################################################
            # Train G_xy and G_yx.
            ################################################################## 
            # Stage 1: x -> y -> x
            optim_g.zero_grad()
            y_fake = G_xy(x)
            pred_y = D_y(y_fake)
            loss_g = criterion(pred_y, labels_real)
            
            x_recon = G_yx(y_fake)
            pred_x = D_x(x_recon)
            loss_g_cycle = criterion(pred_x, labels_real)
            loss_g_total = loss_g + loss_g_cycle
            stats['g_xyx'].append(loss_g_total.item())
            loss_g_total.backward()
            optim_g.step()
            
            # Stage2: y -> x -> y
            optim_g.zero_grad()
            x_fake = G_yx(y)
            pred_x = D_x(x_fake)
            loss_g = criterion(pred_x, labels_real)
            
            y_recon = G_xy(x_fake)
            pred_y = D_y(y_recon)
            loss_g_cycle = criterion(pred_y, labels_real)
            loss_g_total = loss_g + loss_g_cycle
            stats['g_yxy'].append(loss_g_total.item())
            loss_g_total.backward()
            optim_g.step()
        
        # Generate samples to save at specified intervals.
        if epoch % sample_freq == 0:
            sample_x = G_yx(y).detach()
            sample_y = G_yx(x).detach()
            vutils.save_image(sample_x, f'{sample_dir_x}/{epoch+1}.png', 
                              normalize=True)
            vutils.save_image(sample_y, f'{sample_dir_y}/{epoch+1}.png', 
                              normalize=True)
            
        # If specified, save weights corresponding to generated samples.
        if save_weights:
            states = dict(g_xy=G_xy.state_dict(),
                          g_yx=G_yx.state_dict(),
                          dx=D_x.state_dict(),
                          dy=D_y.state_dict(),
                          epoch=epoch+1)
            torch.save(states, f'{sample_dir}/{epoch+1}.pth') 
                
        # Print results for last mini batch of epoch.
        if epoch % print_freq == 0:
            print(f'Epoch [{epoch+1}/{epochs}])')
            print(f"D real loss: {loss_d_real:.3f}\t"
                  f"D fake loss: {loss_d_fake:.3f}")
            print(f"G xyx loss: {stats['g_xyx'][-1]:.3f}\t"
                  f"G yxy fake: {stats['g_yxy'][-1]:.3f}\n")
            
    torch.save(stats, f'{sample_dir}/stats.pkl')
    return stats
