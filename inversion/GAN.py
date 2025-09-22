
import torch

import torch.nn as nn

# Generator Code

# See https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class DCGenerator(nn.Module):
    def __init__(self, input_size=2048, base_features=64, name_suffix=None):
        super(DCGenerator, self).__init__()
        if name_suffix is None:
            self.model_name = f"DCGenerator_{input_size}_{base_features}"
        else:
            self.model_name = f"DCGenerator_{input_size}_{base_features}_{name_suffix}"
        self.nz = input_size
        self.ngf = base_features
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        # input is Z, going into a convolution
        self.deconv1 = nn.ConvTranspose2d(in_channels=input_size, out_channels=base_features * 8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(base_features * 8)
        # state size. ``(ngf*8) x 4 x 4``
        self.deconv2 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_features * 4)
        # state size. ``(ngf*4) x 8 x 8``
        self.deconv3 = nn.ConvTranspose2d( base_features * 4, base_features * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_features * 2)
        # state size. ``(ngf*2) x 16 x 16``
        self.deconv4 = nn.ConvTranspose2d( base_features * 2, base_features, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(base_features)
        # state size. ``(ngf) x 32 x 32``
        self.deconv5 = nn.ConvTranspose2d( base_features, 3, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(3)
        # state size. ``3 x 64 x 64``
        self.ds = torch.as_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

    def forward(self, x):
        x = x.view(-1, self.nz, 1, 1)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.deconv5(x)
        x = self.tanh(x)
        x = self.bn5(x)

        #x = x / self.ds

        return x

    def to(self, *args, **kwargs):
        nw = super(DCGenerator, self).to(*args, **kwargs)
        self.ds = self.ds.to(*args, **kwargs)
        return nw

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                if name == 'deconv1.weight' and param.size(0) != self.deconv1.in_channels:
                    # Adjust the in_channels of deconv1
                    in_channels = self.deconv1.in_channels
                    if param.size(0) > in_channels:
                        print(f'Warning: loaded state_dict has more channels ({param.size(0)}) than the model ({in_channels})')
                        own_state[name].copy_(param[:in_channels, :, :, :])
                    else:
                        print(f'Warning: loaded state_dict has fewer channels ({param.size(0)}) than the model ({in_channels})')
                        own_state[name][:param.size(0), :, :, :].copy_(param)
                        nn.init.kaiming_normal_(own_state[name][param.size(0):, :, :, :], mode='fan_out', nonlinearity='relu')
                else:
                    own_state[name].copy_(param)
            elif strict:
                raise KeyError(f'unexpected key "{name}" in state_dict')
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError(f'missing keys in state_dict: "{missing}"')
    

class DCDiscriminator(nn.Module):
    def __init__(self, base_features=64, name_suffix=None):
        super(DCDiscriminator, self).__init__()
        if name_suffix is None:
            self.model_name = f"DCDiscriminator_{base_features}"
        else:
            self.model_name = f"DCDiscriminator_{base_features}_{name_suffix}"
        self.conv1 = nn.Conv2d(3, base_features, 4, 2, 1, bias=False)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(base_features, base_features * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_features * 2)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(base_features * 2, base_features * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_features * 4)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(base_features * 4, base_features * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(base_features * 8)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(base_features * 8, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu4(x)
        x = self.conv5(x)
        #x = self.sigmoid(x)
        return x