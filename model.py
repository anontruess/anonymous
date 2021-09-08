import torch
import torch.nn as nn
import torch.nn.functional as F
from base_resnet import resnet50, resnet18
import os

class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            print(name)
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if name == 'fc':
                continue
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x): 
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

def load_net(networks, fdir, name):
    net = networks

    filepath = os.path.join(fdir, name)
    print("Loading file...", filepath)

    if not os.path.isfile(filepath):
        print("Checkpoint file" + filepath + " not found!")
        raise IOError

    # print('=> loading checkpoint "{}"'.format(filepath))
    checkpoint_1 = torch.load(filepath)

    net.load_state_dict(checkpoint_1)
    print("=> loaded checkpoint '{}'".format(filepath))
    return net
