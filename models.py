import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(strided, c_in, c_out, f, stride, pad, bias=False, bn=True):
    """Create a conv or deconv block (the latter referring to a backward
    strided convolution) optionally followed by a batch norm layer.
    """
    if strided:
        conv = nn.ConvTranspose2d(c_in, c_out, f, stride, pad, bias=bias)
    else:
        conv = nn.Conv2d(c_in, c_out, f, stride, pad, bias=bias)
    conv.weight.data.normal_(0.0, 0.02)
    layers = [conv]
    if bn:
        bn_ = nn.BatchNorm2d(c_out)
        bn_.weight.data.normal_(1.0, 0.02)
        bn_.bias.data.zero_()
        layers.append(bn_)
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """Residual block to be used in CycleGenerator. Note that the relu or
    leaky must still be applied on the output.
    """

    def __init__(self, c_in, num_layers=2, leak=.02):
        """
        Parameters
        -----------
        c_in: int
            # of input channels.
        num_layers: int
            Number of conv blocks inside the skip connection (default 2).
            ResNet paper notes that skipping a single layer did not show
            noticeable improvements.
        """
        super().__init__()
        self.leak = leak
        self.layers = nn.ModuleList([conv_block(False, c_in, c_in, 3, 1, 1)
                                     for i in range(num_layers)])

    def forward(self, x):
        x_out = x
        for layer in self.layers:
            x_out = F.leaky_relu(layer(x_out), self.leak)
        return x + x_out


class BaseModel(nn.Module):
    """Base model that adds several functions to nn.Module
    for convenience and diagnostics. TESTING
    """

    def __init__(self):
        super().__init__()

    def dims(self):
        """Get shape of each layer's weights."""
        return [p.shape for p in self.parameters()]

    def trainable(self):
        """Check which layers are trainable."""
        return [p.requires_grad for p in self.parameters()]

    def layer_stats(self):
        """Check mean and standard deviation of each layer's weights."""
        return [(round(p.data.mean().item(), 3),
                 round(p.data.std().item(), 3))
                for p in self.parameters()]

    def plot_weights(self):
        n_layers = len(self.dims())
        fig, ax = plt.subplots(n_layers, figsize=(8, n_layers * 1.25))
        for i, p in enumerate(self.parameters()):
            ax[i].hist(p.data.flatten())
            ax[i].set_title(p.shape)
        plt.tight_layout()
        plt.show()


class Generator(BaseModel):
    """DCGAN Generator"""
    
    def __init__(self, input_c=100, img_c=3, ngf=64):
        """
        Parameters
        -----------
        input_c: int
            Depth of random noise input into model.
        img_c: int
            # of channels of output image.
        ngf: int
            # of filters to output in last deconv block before the final
            reduction to img_c channels.
        """
        super().__init__()
        # 100 x 1 x 1 -> 512 x 4 x 4
        deconv1 = conv_block(True, input_c, ngf*8, f=4, stride=1, pad=0)
        # 512 x 4 x 4 -> 256 x 8 x 8
        deconv2 = conv_block(True, ngf*8, ngf*4, 4, 2, 1)
        # 256 x 8 x 8 -> 128 x 16 x 16
        deconv3 = conv_block(True, ngf*4, ngf*2, 4, 2, 1)
        # 128 x 16 x 16 -> 64 x 32 x 32
        deconv4 = conv_block(True, ngf*2, ngf, 4, 2, 1)
        # 64 x 32 x 32 -> 3 x 64 x 64
        deconv5 = conv_block(True, ngf, img_c, 4, 2, 1, bn=False)

        self.layers = nn.ModuleList([deconv1, deconv2, deconv3, deconv4,
                                     deconv5])
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x), inplace=True)
        x = torch.tanh(self.layers[-1](x))
        return x

    
class Discriminator(BaseModel):
    """DCGAN discriminator."""
    
    def __init__(self, ndf=64, img_c=3, leak=.02):
        """
        Parameters
        -----------
        ndf: int
            # of filters in first conv layer.
        img_c: int
            # of channels in input image.
        leak: float
            Slope of leaky relu where x < 0.
        """
        super().__init__()
        self.leak = leak

        # Dimensions for default values (most inputs resized to 3 x 64 x 64).
        # 3 x 64 x 64 -> 64 x 32 x 32
        conv1 = conv_block(False, img_c, ndf, f=4, stride=2, pad=1, bn=False)
        # 64 x 32 x 32 -> 128 x 16 x 16
        conv2 = conv_block(False, ndf, ndf*2, 4, 2, 1)
        # 128 x 16 x 16 -> 256 x 8 x 8
        conv3 = conv_block(False, ndf*2, ndf*4, 4, 2, 1)
        # 256 x 8 x 8 -> 512 x 4 x 4
        conv4 = conv_block(False, ndf*4, ndf*8, 4, 2, 1)
        # 512 x 4 x 4 -> 1 x 1 x 1
        conv5 = conv_block(False, ndf*8, 1, 4, 1, 0, bn=False)

        self.layers = nn.ModuleList([conv1, conv2, conv3, conv4, conv5])
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x), self.leak, inplace=True)
        x = torch.sigmoid(self.layers[-1](x))
        return x.squeeze()


class CycleGenerator(BaseModel):
    """CycleGAN Generator."""

    def __init__(self, img_c=3, ngf=64, leak=.02):
        """
        Parameters
        -----------
        img_c: int
            # of channels of input image.
        ngf: int
            # of channels in first convolutional layer.
        leak: float
            Slope of leaky relu where x < 0. Leak of 0 is regular relu.
        """
        super().__init__()
        self.leak = leak
        self.activation = nn.LeakyReLU(self.leak)

        # ENCODER
        # 3 x 64 x 64 -> 64 x 32 x 32
        deconv1 = conv_block(False, img_c, ngf, f=4, stride=2, pad=1)
        # 64 x 32 x 32 -> 128 x 16 x 16
        deconv2 = conv_block(False, ngf, ngf*2, 4, 2, 1)
        self.encoder = nn.Sequential(deconv1,
                                     self.activation,
                                     deconv2,
                                     self.activation)

        # TRANSFORMER
        # 128 x 16 x 16 -> 128 x 16 x 16
        res1 = ResBlock(ngf*2, num_layers=2, leak=self.leak)
        # 128 x 16 x 16 -> 128 x 16 x 16
        res2 = ResBlock(ngf*2, 2, self.leak)
        self.transformer = nn.Sequential(res1,
                                         self.activation,
                                         res2,
                                         self.activation)

        # DECODER
        # 128 x 16 x 16 -> 64 x 32 x 32
        deconv1 = conv_block(True, ngf*2, ngf, f=4, stride=2, pad=1)
        # 64 x 32 x 32 -> 3 x 64 x 64
        deconv2 = conv_block(True, ngf, img_c, 4, 2, 1)
        self.decoder = nn.Sequential(deconv1,
                                     self.activation,
                                     deconv2,
                                     nn.Tanh())

        # Module list of Sequential objects is helpful if we want to use
        # different learning rates per group.
        self.groups = nn.ModuleList([self.encoder,
                                     self.transformer,
                                     self.decoder])

    def forward(self, x):
        for group in self.groups:
            x = group(x)
        return x
