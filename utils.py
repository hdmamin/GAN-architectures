import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from pathlib import Path
import torchvision.utils as vutils


def show_batch(dl, limit=64, size=10):
    """Display a batch of images.
    
    Parameters
    -----------
    dl: DataLoader
    limit: int
        Max # of images to display (since we usually don't need to see all 128
        images in the batch).
    size: int
        Height and width of the square grid of images displayed.
    """
    batch = next(iter(dl))[0]
    fig, ax = plt.subplots(figsize=(size, size))
    grid = vutils.make_grid(batch[:limit], normalize=True, nrow=8)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis('off')
    plt.show()
    
    
def save_real_batch(dl, path='samples'):
    """Saves a real batch of images in the same grid format as the generated
    images for easy comparison.
      
    Parameters
    -----------
    dl: DataLoader
    path: str
        Directory to save image grid in.
    """
    batch = next(iter(dl))
    vutils.save_image(batch[0], f'{path}/real_batch.png', normalize=True)


def plot_output(output):
    """Plot losses and average predictions by mini batch. Pass in output of
    DCGAN train function.
    """
    # Plot losses by mini batch.
    fig, ax = plt.subplots(2, 1, figsize=(9, 6))
    ax[0].plot(output['d_real_loss'], label='D real')
    ax[0].plot(output['d_fake_loss'], label='D fake')
    ax[0].plot(output['g_loss'], label='G')
    ax[0].set_title('Loss by Mini Batch')
    ax[0].legend()
    
    # Plot soft prediction average by mini batch.
    ax[1].plot(output['d_real_avg'], label='Real examples')
    ax[1].plot(output['d_fake_avg'], label='Fake examples')
    ax[1].set_title('Average Soft Predictions (D) by Mini Batch')
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    
def read_img(path):
    """Load image using openCV and convert to RGB color space."""
    path = str(path)
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    
def show_img(path):
    """Load image from path and display."""
    path = str(path)
    img = read_img(path)
    fig = plt.figure(figsize=(9, 9))
    plt.imshow(img)
    plt.axis('off')
    name = path.split('/')[-1]
    plt.title(name)
    plt.show()


def sorted_paths(dir_):
    """Pass in the name of a directory as a string and return a list of
    file Path objects sorted by epoch number.
    """
    return sorted([f for f in Path(dir_).iterdir() if str(f.parts[-1])[0].isnumeric()],
                  key=lambda x: int(x.parts[-1].split('.')[0]))


def show_samples(sample_dir='samples'):
    """Show samples from each """
    for f in sorted_paths(sample_dir):
        show_img(f)


def render_samples(path):
    """Render animation from all files in a directory of samples.

    Parameters
    -----------
    path: str
        Directory that contains samples images.
    out_files: str
        Location to save new file.
    """
    out_file = path + '.gif'
    matplotlib.rcParams['animation.convert_path'] = 'magick'
    fig = plt.figure(figsize=(9, 9))
    plt.axis('off')
    arrs = [read_img(file) for file in sorted_paths(path)]
    plots = [[plt.imshow(arr, animated=True)] for arr in arrs]
    anim = animation.ArtistAnimation(fig, plots, blit=True, repeat_delay=500,
                                     repeat=True)
    print(f'Writing file to {out_file}')
    anim.save(out_file, writer='imagemagick', fps=2)


def stats(x):
    """Quick wrapper to get mean and standard deviation of a tensor."""
    return round(x.mean().item(), 4), round(x.std().item(), 4)


def coords2img(row, out_file, out_dir='cat_data'):
    """Convert the line coordinates, stored as a string in the raw quickdraw
    dataset, into an image and save it in the output directory.
    
    Parameters
    -----------
    row: str
        One item in our df/series corresponding to a single image.
    out_file: str
        File name of output.
    out_dir: str
        Name of output directory.
    """
    output_file = os.path.join(out_dir, f'{out_file}.png')
    lines = [np.array(line)/255 for line in ast.literal_eval(row)]
    # Save as 64px x 64px img. Flip y axis because 0,0 is given as top left.
    fig, ax = plt.subplots(figsize=(.64, .64), dpi=100)
    plt.gca().invert_yaxis()
    for i, line in enumerate(lines):
        ax.plot(line[0, :], line[1, :], '-', c='black')
    plt.axis('off')
    plt.savefig(output_file, dpi=100)
    plt.close()
    

def gen_images(data, start=0, subset=None):
    """Generate images for all examples in our quickdraw dataset.
    
    Parameters
    -----------
    data: pd.Series
        Contains stroke coordinatess for all images.
    start: int
        Index of file to start with (useful if memory error forces us to 
        resume in the middle - leave as zero to start from the beginning).
    subset: int
        Limit the number of samples to create (useful for testing - leave as
        None to create all samples).
    """
    for i, row in enumerate(data[start:subset], start):
        if i % 1000 == 0: print(f'Processing file {i}...')
        coords2img(row, i)