import torch
import torch.nn.functional as F
import torchvision.utils as vutils


def train(epochs, dl, lr=2e-4, b1=.5, sample_freq=10, sample_dir='samples',
          weight_dir=None, d=None, g=None):
    """Train generator and discriminator with Adam.
    
    Parameters
    -----------
    epochs: int
        Number of epochs to train for.
    dl: DataLoader
    lr: float
        Learning rate. Paper recommends .0002.
    beta1: float
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
        g.train()
        d.train()
    
    # Define loss and optimizers.
    criterion = nn.BCELoss()
    d_optim = torch.optim.Adam(d.parameters(), lr=lr, betas=(b1, .999))
    g_optim = torch.optim.Adam(g.parameters(), lr=lr, betas=(b1, .999))
    
    # Noise used for sample images, not training.
    fixed_noise = torch.randn(dl.batch_size, 100, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    
    # Store stats to return at end.
    d_real_losses = []
    d_fake_losses = []
    d_real_avg = []
    d_fake_avg = []
    g_losses = []
    samples = []
    
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
            d_fake_losses.append(d_loss_fake)
            d_real_losses.append(d_loss_real)
            
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
            g_losses.append(g_loss)
            g_loss.backward()
            g_optim.step()
        
        # Print losses.
        print(f'Epoch [{epoch+1}/{epochs}] \nBatch {i+1} Metrics:')
        print(f'D loss (real): {d_loss_real:.4f}\t', end='')
        print(f'D loss (fake): {d_loss_fake:.4f}')
        print(f'G loss: {g_loss:.4f}\n')

        # Generate sample from fixed noise at end of every fifth epoch.
        if epoch % sample_freq == 0:
            with torch.no_grad():
                fake = g(fixed_noise)
                vutils.save_image(fake.detach(), 
                                  f'{sample_dir}/fake_epoch_{epoch}.png',
                                  normalize=True)
                
            # If specified, save weights corresponding to generated samples.
            if weight_dir:
                states = dict(g=g.state_dict(), 
                              d=d.state_dict(),
                              g_optimizer=g_optim.state_dict(),
                              d_optimizer=d_optim.state_dict(),
                              epoch=epoch)
                torch.save(states, f'{weight_dir}/model_{epoch}.pth')
        
    return dict(d_real_losses=d_real_losses, 
                d_fake_losses=d_fake_losses,
                g_losses=g_losses,
                d_real_avg=d_real_avg,
                d_fake_avg=d_fake_avg)