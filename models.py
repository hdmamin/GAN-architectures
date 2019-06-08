import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(strided, c_in, c_out, f, stride, pad, bias=False, bn=True):
    """Create deconv block consisting of a backward strided convolution 
    optionally followed by a batch norm layer.
    """
    if strided:
        conv_ = nn.ConvTranspose2d(c_in, c_out, f, stride, pad, bias=bias)
    else:
        conv_ = nn.Conv2d(c_in, c_out, f, stride, pad, bias=bias)
    conv_.weight.data.normal_(0.0, 0.02)
    layers = [conv_]
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
        fig, ax = plt.subplots()
        for p in self.parameters():
            ax[][].hist(p.weight.data)
            ax[][].set_title()
        plt.show()
    
class Generator(BaseModel):
    """DCGAN Generator"""
    
    def __init__(self, input_c=100, final_c=64):
        super().__init__()
        # 100 x 1 x 1 -> 512 x 4 x 4
        self.deconv1 = conv(True, input_c, final_c*8, f=4, stride=1, pad=0)
        # 512 x 4 x 4 -> 256 x 8 x 8
        self.deconv2 = conv(True, final_c*8, final_c*4, 4, 2, 1)
        # 256 x 8 x 8 -> 128 x 16 x 16
        self.deconv3 = conv(True, final_c*4, final_c*2, 4, 2, 1)
        # 128 x 16 x 16 -> 64 x 32 x 32
        self.deconv4 = conv(True, final_c*2, final_c, 4, 2, 1)
        # 64 x 32 x 32 -> 3 x 64 x 64
        self.deconv5 = conv(True, final_c, 3, 4, 2, 1, bn=False)
        
    def forward(self, x):
        x = F.relu(self.deconv1(x), True)
        x = F.relu(self.deconv2(x), True)
        x = F.relu(self.deconv3(x), True)
        x = F.relu(self.deconv4(x), True)
        x = torch.tanh(self.deconv5(x))
        return x

    
class Discriminator(BaseModel):
    """DCGAN discriminator."""
    
    def __init__(self, dim_1=64, leak=.02):
        """
        Parameters
        -----------
        dim_1: int
            # of filters in first conv layer.
        leak: float
            Slope of leaky relu where x < 0.
        """
        super().__init__()
        self.leak = leak
        
        # 3 x 64 x 64 -> 64 x 32 x 32
        self.conv1 = conv(False, 3, dim_1, f=4, stride=2, pad=1, bn=False)
        # 64 x 32 x 32 -> 128 x 16 x 16
        self.conv2 = conv(False, dim_1, dim_1*2, 4, 2, 1)
        # 128 x 16 x 16 -> 256 x 8 x 8
        self.conv3 = conv(False, dim_1*2, dim_1*4, 4, 2, 1)
        # 256 x 8 x 8 -> 512 x 4 x 4
        self.conv4 = conv(False, dim_1*4, dim_1*8, 4, 2, 1)
        # 512 x 4 x 4 -> 1 x 1 x 1
        self.conv5 = conv(False, dim_1*8, 1, 4, 1, 0, bn=False)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), self.leak, True)
        x = F.leaky_relu(self.conv2(x), self.leak, True)
        x = F.leaky_relu(self.conv3(x), self.leak, True)
        x = F.leaky_relu(self.conv4(x), self.leak, True)
        x = torch.sigmoid(self.conv5(x))
        return x.squeeze()
