import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from pathlib import Path
from skimage.io import imread
import torchvision.utils as vutils


def show_batch(dl, limit=64):
    """Display a batch of images.
    
    Parameters
    -----------
    dl: DataLoader
    limit: int
        Max # of images to display (since we usually don't need to see all 128
        images in the batch).
    """
    batch = next(iter(dl))
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(np.transpose(vutils.make_grid(batch[0][:limit], normalize=True, 
                                             nrow=8), (1, 2, 0)))
    plt.axis('off')
    plt.show()


def plot_output(output):
    """Plot losses and average predictions by mini batch."""
    # Plot losses by mini batch.
    fig, ax = plt.subplots(2, 1, figsize=(9, 6))
    ax[0].plot(output['d_real_losses'], label='D real')
    ax[0].plot(output['d_fake_losses'], label='D fake')
    ax[0].plot(output['g_losses'], label='G')
    ax[0].set_title('Loss by Mini Batch')
    ax[0].legend()
    
    # Plot soft prediction average by mini batch.
    ax[1].plot(output['d_real_avg'], label='Real examples')
    ax[1].plot(output['d_fake_avg'], label='Fake examples')
    ax[1].set_title('Average Soft Predictions (D) by Mini Batch')
    plt.tight_layout()
    plt.legend()
    plt.show()


def show_img(path):
    """Load image from path and display."""
    path = str(path)
    img = imread(path)
    fig = plt.figure(figsize=(9, 9))
    plt.imshow(img)
    if 'epoch' in path:
        num = path.split('.')[0].split('epoch_')[-1]
        plt.title(f'Epoch {num}')
    plt.show()


def show_samples(sample_dir='samples'):
    """Show samples from each """
    pattern = 'fake_epoch_'
    files = list(Path(sample_dir).glob(pattern+'*'))
    for f in sorted(files, 
                    key=lambda x: int(str(x).split('.')[-2].split('_')[-1])):
        show_img(f)

        
def render_samples(path, out_file):
    """Render animation from all files in a directory of samples.
    
    Parameters
    -----------
    path: str
        Directory that contains samples images.
    out_files: str
        Location to save new file.
    """
    matplotlib.rcParams['animation.convert_path'] = 'magick'
    fig = plt.figure(figsize=(9, 9))
    plt.axis('off')
    arrs = [cv2.imread(str(file)) for file in Path(path).iterdir()]
    plots = [[plt.imshow(arr, animated=True)] for arr in arrs]
    anim = animation.ArtistAnimation(fig, plots, blit=True, repeat_delay=10, 
                                     repeat=True)
    print(f'Writing file to {out_file}')
    anim.save(out_file, writer='imagemagick', fps=2)