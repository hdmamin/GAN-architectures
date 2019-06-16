import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder

from config import *


class DaVinciDS(Dataset):

    def __init__(self, path, size, tfms=False):
        """
        Parameters
        -----------
        tfms: list
            Transforms, not including normalize.
        """
        super().__init__()
        self.files = [str(file) for file in list(Path(path).iterdir())
                      if 'gt' not in str(file)]
        self.tfms = tfms
        self.size = size

    def __getitem__(self, i):
        """Y does not matter here, just return placeholder to work with
        our training loop.
        """
        img = cv2.cvtColor(cv2.imread(self.files[i]), cv2.COLOR_BGR2RGB)
        if self.tfms:
            img = self.resize(img)
        img = self.normalize(img / 255)
        return np.moveaxis(img.astype(np.float32), -1, 0), 1

    def __len__(self):
        return len(self.files)

    def resize(self, img):
        w, h, _ = img.shape
        return cv2.resize(img, (self.size, self.size))

    def normalize(self, img):
        stats = np.array([[.5, .5, .5],
                          [.5, .5, .5]])
        return (img - stats[0]) / stats[1]


# Davinci transforms and dataset.
dv_tfms = transforms.Compose([transforms.Resize(img_size),
                              transforms.ToTensor()])
dv_ds = DaVinciDS(root_dv, img_size, dv_tfms)
dv_dl = DataLoader(dv_ds, batch_size=36, shuffle=True, num_workers=workers)

# Transforms for photo and sketch datasets.
tfms = transforms.Compose([transforms.Resize(img_size),
                           transforms.ToTensor(),
                           transforms.Normalize([.5, .5, .5], [.5, .5, .5])])

# Photo dataset
photo_ds = ImageFolder(root_photo, transform=tfms)
photo_dl = DataLoader(photo_ds, batch_size=bs, shuffle=True,
                      num_workers=workers)

# Sketch dataset
sketch_ds = ImageFolder(root_sketch, transform=tfms)
sketch_dl = DataLoader(sketch_ds, batch_size=bs, shuffle=True,
                       num_workers=workers)

# Small photo dataset for testing: camel, giraffe, horse, zebra
small_ds = ImageFolder(root_small, transform=tfms)
small_dl = DataLoader(small_ds, batch_size=bs, shuffle=True,
                      num_workers=workers)

# CIFAR10 dataset (must download the first time we run code)
cifar_ds = datasets.CIFAR10(root='cifar',
                            download=True,
                            transform=tfms)
cifar_dl = DataLoader(cifar_ds, batch_size=bs, shuffle=True,
                      num_workers=workers)

# MNIST transforms and dataset (must download the first time we run code)
# mnist_tfms = transforms.Compose([transforms.Resize(img_size),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize((0.1307,), (0.3081,))])
# mnist_ds = datasets.MNIST(root='mnist',
#                           download=False,
#                           train=True,
#                           transform=mnist_tfms)
# mnist_dl = DataLoader(mnist_ds, batch_size=bs, shuffle=True,
#                       num_workers=workers)

# Celeb A dataset. Non-square images require additional cropping step.
celeb_tfms = transforms.Compose([transforms.Resize(img_size),
                                 transforms.CenterCrop(img_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize([.5, .5, .5],
                                                      [.5, .5, .5])])
celeb_ds = ImageFolder(root_celeb, transform=celeb_tfms)
celeb_dl = DataLoader(celeb_ds, batch_size=bs, shuffle=True, num_workers=workers)
