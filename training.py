from collections import defaultdict
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from models import Discriminator, Generator, CycleGenerator
from config import *


def NEW_train(epochs, dl, lr=2e-4, b1=.5, sample_freq=10, sample_dir='samples',
              weight_dir=None, d=None, g=None, d_head_start=0, gd_ratio=1):
    """Train generator and discriminator with Adam.
    
    NOTE: SOME CHANGES IN TRAIN MAY NOT BE REFLECTED HERE - CHECK BEFORE USING.
    ALSO, GAN HACKS SUGGESTS TRYING TO TUNE GD_RATIO IS A HOPELESS TASK.
    
    IN PROGRESS - getting errors suggesting I need retain_graph=True in 
    G loss.backward()? Seems to be caused by d_head_start. Didn't get to 
    dg_ratio yet, so that could cause the same issue. Not sure why this
    is happening - if G is not being trained for the first few epochs, 
    then we haven't called g_loss.backward() yet. Also look into whether
    to alternate epochs or mini batches - maybe epochs would be good
    to give a little more time for one to train? Or will that result
    in one overpowering the other?
    
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
    weight_dir: str
        Directory to save weights with sample frequency as samples are
        generated. Default is None, meaning no weights will be saved.
    d: nn.Module
        Discriminator (binary classifier, 1 for real, 0 for fake).
    g: nn.Module
        Generator (upsamples random noise into image).
    d_head_start: int
        Number of epochs to train D for before starting to train G.
    gd_ratio: int
        Number of times to train G for each time we train D. In other words,
        G will be trained every epoch, but D will only be trained every
        gd_ratio epochs.
    """
    if not (d or g):
        g = Generator().to(device)
        d = Discriminator().to(device)
    # For GANs, models should stay in train mode.
    g.train()
    d.train()

    # Define loss and optimizers.
    criterion = nn.BCELoss()
    d_optim = torch.optim.Adam(d.parameters(), lr=lr, betas=(b1, .999))
    g_optim = torch.optim.Adam(g.parameters(), lr=lr, betas=(b1, .999))

    # Noise used for sample images, not training.
    fixed_noise = torch.randn(dl.batch_size, 100, 1, 1, device=device)

    # Store stats to return at end.
    d_real_losses = []
    d_fake_losses = []
    d_real_avg = []
    d_fake_avg = []
    g_losses = []

    # Suppress printed output to 20 total times - think it's slowing down nb.
    print_freq = 1
    if quiet_mode:
        print_freq = max(1, epochs // 20)

    # Train D and G.
    for epoch in range(epochs):
        for i, (x, y) in enumerate(dl):
            x = x.to(device)
            bs_curr = x.shape[0]
            real_labels = torch.ones(bs_curr, device=device)
            fake_labels = torch.zeros(bs_curr, device=device)
            noise = torch.randn(bs_curr, 100, 1, 1, device=device)
            train_d = (i % gd_ratio == 0) or (epoch < d_head_start)
            train_g = (epoch >= d_head_start)

            ##################################################################
            # Train discriminator. Detach G output for speed.
            ##################################################################
            d_optim.zero_grad()
            if train_d:
                fake = g(noise)
                y_hat_fake = d(fake.detach())
                y_hat_real = d(x)

                # Compute losses.
                d_loss_fake = criterion(y_hat_fake, fake_labels)
                d_loss_real = criterion(y_hat_real, real_labels)
                d_loss = d_loss_real + d_loss_fake

                # Update stats.
                fake_avg = y_hat_fake.mean().item()
                real_avg = y_hat_real.mean().item()
                d_fake_avg.append(fake_avg)
                d_real_avg.append(real_avg)
                d_fake_losses.append(d_loss_fake.item())
                d_real_losses.append(d_loss_real.item())

                # Backpropagation.
                d_loss.backward()
                d_optim.step()

            ##################################################################
            # Train generator. Use true_labels because we want to fool D.
            # G's weights are the same so no need to re-generate fake
            # examples. D was updated so compute loss again.
            ##################################################################
            g_optim.zero_grad()
            if train_g:
                g_loss = criterion(d(fake), real_labels)
                g_losses.append(g_loss.item())
                g_loss.backward()
                g_optim.step()

        # Print losses.
        if epoch % print_freq == 0:
            print(f'Epoch [{epoch+1}/{epochs}] \nBatch {i+1} Metrics:')
            if train_d:
                print(f'D loss (real): {d_loss_real:.4f}\t', end='')
                print(f'D loss (fake): {d_loss_fake:.4f}')
            if train_g:
                print(f'G loss: {g_loss:.4f}\n')

        # Generate sample from fixed noise at end of every fifth epoch.
        if epoch % sample_freq == 0:
            with torch.no_grad():
                fake = g(fixed_noise)
                vutils.save_image(fake.detach(), 
                                  f'{sample_dir}/{epoch}.png',
                                  normalize=True)
                
            # If specified, save weights corresponding to generated samples.
            if weight_dir:
                states = dict(g=g.state_dict(), 
                              d=d.state_dict(),
                              g_optimizer=g_optim.state_dict(),
                              d_optimizer=d_optim.state_dict(),
                              epoch=epoch)
                torch.save(states, f'{weight_dir}/{epoch}.pth')
        
    return dict(d_real_losses=d_real_losses, 
                d_fake_losses=d_fake_losses,
                g_losses=g_losses,
                d_real_avg=d_real_avg,
                d_fake_avg=d_fake_avg)


def train(epochs, dl, lr=2e-4, b1=.5, sample_freq=10, quiet_mode=True,
          sample_dir='samples', weight_dir=None, d=None, g=None):
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
    weight_dir: str
        Directory to save weights with sample frequency as samples are
        generated. Default is None, meaning no weights will be saved.
    d: nn.Module
        Discriminator (binary classifier, 1 for real, 0 for fake).
    g: nn.Module
        Generator (upsamples random noise into image).
    """
    if not (d or g):
        g = Generator().to(device)
        d = Discriminator().to(device)
    # For GANs, models should stay in train mode.
    g.train()
    d.train()
    
    # Define loss and optimizers.
    criterion = nn.BCELoss()
    d_optim = torch.optim.Adam(d.parameters(), lr=lr, betas=(b1, .999))
    g_optim = torch.optim.Adam(g.parameters(), lr=lr, betas=(b1, .999))
    
    # Noise used for sample images, not training.
    fixed_noise = torch.randn(dl.batch_size, 100, 1, 1, device=device)

    # Store stats to return at end.
    d_real_losses = []
    d_fake_losses = []
    d_real_avg = []
    d_fake_avg = []
    g_losses = []

    # Suppress printed output to 20 total times - think it's slowing down nb.
    print_freq = 1
    if quiet_mode:
        print_freq = max(1, epochs // 20)
        
    # Train D and G.
    for epoch in range(epochs):
        for i, (x, y) in enumerate(dl):
            x = x.to(device)
            bs_curr = x.shape[0]
            real_labels = torch.ones(bs_curr, device=device)
            fake_labels = torch.zeros(bs_curr, device=device)
            noise = torch.randn(bs_curr, 100, 1, 1, device=device)
            
            ##################################################################
            # Train discriminator. Detach G output for speed.
            ##################################################################
            d_optim.zero_grad()
            fake = g(noise)
            y_hat_fake = d(fake.detach())
            y_hat_real = d(x)
            
            # Compute losses.
            d_loss_fake = criterion(y_hat_fake, fake_labels)
            d_loss_real = criterion(y_hat_real, real_labels)
            d_loss = d_loss_real + d_loss_fake
            
            # Update stats.
            fake_avg = y_hat_fake.mean().item()
            real_avg = y_hat_real.mean().item()
            d_fake_avg.append(fake_avg)
            d_real_avg.append(real_avg)
            d_fake_losses.append(d_loss_fake.item())
            d_real_losses.append(d_loss_real.item())
            
            # Backpropagation.
            d_loss.backward()
            d_optim.step()
            
            ##################################################################
            # Train generator. Use true_labels because we want to fool D.
            # G's weights are the same so no need to re-generate fake 
            # examples. D was updated so compute loss again. 
            ##################################################################
            g_optim.zero_grad()
            g_loss = criterion(d(fake), real_labels)
            g_losses.append(g_loss.item())
            g_loss.backward()
            g_optim.step()
        
        # Print losses.
        if epoch % print_freq == 0:
            print(f'Epoch [{epoch+1}/{epochs}] \nBatch {i+1} Metrics:')
            print(f'D loss (real): {d_loss_real:.4f}\t', end='')
            print(f'D loss (fake): {d_loss_fake:.4f}')
            print(f'G loss: {g_loss:.4f}\n')

        # Generate sample from fixed noise at end of every fifth epoch.
        if epoch % sample_freq == 0:
            with torch.no_grad():
                fake = g(fixed_noise)
                vutils.save_image(fake.detach(), 
                                  f'{sample_dir}/{epoch}.png',
                                  normalize=True)
                
            # If specified, save weights corresponding to generated samples.
            if weight_dir:
                states = dict(g=g.state_dict(), 
                              d=d.state_dict(),
                              g_optimizer=g_optim.state_dict(),
                              d_optimizer=d_optim.state_dict(),
                              epoch=epoch)
                torch.save(states, f'{weight_dir}/{epoch}.pth')
        
    return dict(d_real_losses=d_real_losses, 
                d_fake_losses=d_fake_losses,
                g_losses=g_losses,
                d_real_avg=d_real_avg,
                d_fake_avg=d_fake_avg)


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


def train_cycle_gan(epochs, x_dl, y_dl, sample_dir_x, sample_dir_y,
                    weight_dir=None, sample_freq=10, lr=2e-4, b1=.5,
                    loss_type='bce', quiet_mode=True, load_path=None,
                    models=None):
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
    # Create models.
    if not models:
        G_xy, G_yx, D_x, D_y = get_cycle_models(load_path)
    else:
        for model in models:
            model.to(device)
            model.train()
        G_xy, G_yx, D_x, D_y = models
    
    # Create optimizers.
    optim_g = torch.optim.Adam(chain(G_xy.parameters(), G_yx.parameters()), 
                               lr, betas=(b1, .999))
    optim_d = torch.optim.Adam(chain(D_x.parameters(), D_y.parameters()),
                               lr, betas=(b1, .999))    
    
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
    for epoch in range(epochs):
        G_xy.train()
        G_yx.train()
        D_x.train()
        D_y.train()
        
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
            vutils.save_image(sample_x, f'{sample_dir_x}/{epoch}.png', 
                              normalize=True)
            vutils.save_image(sample_y, f'{sample_dir_y}/{epoch}.png', 
                              normalize=True)
            
        # If specified, save weights corresponding to generated samples.
        if weight_dir:
            states = dict(g_xy=G_xy.state_dict(),
                          g_yx=G_yx.state_dict(),
                          dx=D_x.state_dict(),
                          dy=D_y.state_dict(),
                          epoch=epoch)
            torch.save(states, f'{weight_dir}/{epoch}.pth') 
                
        # Print results for last mini batch of epoch.
        if epoch % print_freq == 0:
            print(f'Epoch [{epoch+1}/{epochs}])')
            print(f"D real loss: {loss_d_real:.3f}\t"
                  f"D fake loss: {loss_d_fake:.3f}")
            print(f"G xyx loss: {stats['g_xyx'][-1]:.3f}\t"
                  f"G yxy fake: {stats['g_yxy'][-1]:.3f}\n")
            
    return stats
