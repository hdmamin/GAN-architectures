import torch


root_photo = 'sketch_data/photo/tx_000100000000'
root_sketch = 'sketch_data/sketch/tx_000100000000'
root_dv = 'davinci_data/train'
root_celeb = 'celeba'
root_small = root_photo + '/tmp'  # Contains camel, giraffe, horse, zebra

bs = 64                # Batch size (paper uses 128).
img_size = 64          # Size of input (here it's 64 x 64).
workers = 2            # Number of workers for data loader.
input_c = 100          # Depth of input noise (1 x 1 x noise_dim). AKA nz.
ngf = 64               # Filters in last layer of G before reduced to 3.
ndf = 64               # Filters in first conv layer in D.
img_c = 3              # Channels in input image.
lr = 2e-4              # Recommended learning rate of .0002.
beta1 = .5             # Recommended parameter for Adam.
nc = 3                 # Number of channels of input image.
ngpu = 1               # Number of GPUs to use.
sample_dir = 'samples' # Directory to store sample images from G.
weight_dir = 'weights' # Directory to store model weights.
device = torch.device('cuda:0' if torch.cuda.is_available() and ngpu > 0
                      else 'cpu')
