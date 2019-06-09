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


class BaseModel(nn.Module):
    """Base model that adds several functions to nn.Module
    for convenience and diagnostics.
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
        fig, ax = plt.subplots(len(list(self.parameters)))
        for i, p in enumerate(self.parameters()):
            ax[i].hist(p.weight.data)
            ax[i].set_title(p.shape)
        plt.show()


class Generator(BaseModel):
    """DCGAN Generator"""
    
    def __init__(self, input_c=100, ngf=64):
        super().__init__()
        # 100 x 1 x 1 -> 512 x 4 x 4
        self.deconv1 = conv_block(True, input_c, ngf*8, f=4, stride=1, pad=0)
        # 512 x 4 x 4 -> 256 x 8 x 8
        self.deconv2 = conv_block(True, ngf*8, ngf*4, 4, 2, 1)
        # 256 x 8 x 8 -> 128 x 16 x 16
        self.deconv3 = conv_block(True, ngf*4, ngf*2, 4, 2, 1)
        # 128 x 16 x 16 -> 64 x 32 x 32
        self.deconv4 = conv_block(True, ngf*2, ngf, 4, 2, 1)
        # 64 x 32 x 32 -> 3 x 64 x 64
        self.deconv5 = conv_block(True, ngf, 3, 4, 2, 1, bn=False)
        
    def forward(self, x):
        x = F.relu(self.deconv1(x), True)
        x = F.relu(self.deconv2(x), True)
        x = F.relu(self.deconv3(x), True)
        x = F.relu(self.deconv4(x), True)
        x = torch.tanh(self.deconv5(x))
        return x

    
class Discriminator(BaseModel):
    """DCGAN discriminator."""
    
    def __init__(self, ndf=64, leak=.02):
        """
        Parameters
        -----------
        ndf: int
            # of filters in first conv layer.
        leak: float
            Slope of leaky relu where x < 0.
        """
        super().__init__()
        self.leak = leak

        # 3 x 64 x 64 -> 64 x 32 x 32
        conv1 = conv_block(False, 3, ndf, f=4, stride=2, pad=1, bn=False)
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
            x = F.leaky_relu(layer(x), inplace=True)
        x = torch.sigmoid(self.layers[-1](x))
        return x.squeeze()
