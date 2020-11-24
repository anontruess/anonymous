import argparse
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import random, string
import math
import pickle
from collections import OrderedDict
import torch
from torch import nn as nn, optim as optim
from torch.autograd import Variable

import datetime
from scipy.misc import imsave, imresize

from pdb import set_trace as bp
import powerlaw
from model import Model, Net, load_net

from data_loader import get_train_valid_loader, get_test_loader


parser = argparse.ArgumentParser(description='PyTorch Visualise Filters')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--gpu', default=2, type=int,
                    help='GPU number')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                    help='mini-batch size (default: 4),only used for train')
parser.add_argument('--dataset', default="cifar10", type=str,
                    help='dataset to train on')
parser.add_argument('--layer', default=None, type=str)
parser.add_argument('--seed', default=11, type=int)
parser.add_argument('--split', default=None, required=True, help='train/test')

sampled_GT = None

label_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

def log(f, txt, do_print=1):
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')


def print_graph(maps, title, save_path):
    fig = plt.figure()
    st = fig.suptitle(title)
    for i, (map, args) in enumerate(maps):
        plt.subplot(1, len(maps), i + 1)
        if len(map.shape) > 2 and map.shape[0] == 3:
            plt.imshow(map.transpose((1, 2, 0)).astype(np.uint8),aspect='equal', **args)
        else:
            plt.imshow(map, aspect='equal', **args)
            plt.axis('off')
    plt.savefig(save_path + ".png", bbox_inches='tight', pad_inches = 0)
    fig.clf()
    plt.clf()
    plt.close()

excluded_layers = ['conv4_1','conv5_1','conv6_1','conv7_1']

@torch.no_grad()
def test_function(X, Y, network):
    X = torch.autograd.Variable(torch.from_numpy(X)).cuda()
    Y = torch.autograd.Variable(torch.from_numpy(Y)).cuda()

    network = network.cuda()
    network.eval()
    output = network(X) # (B,1,h,w)
    loss = 0.0

    return

# Visualize feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def plot_mean_activations(filter_save_path, class_wise_activations):

    with open(os.path.join(filter_save_path, args.layer + "_activations.pkl"), 'wb') as f:
        pickle.dump(class_wise_activations, f, protocol=2)

    for cl in class_wise_activations.keys():
        if len(class_wise_activations[cl]):
            class_wise_activations[cl] = np.mean(np.array(class_wise_activations[cl]), axis=0)

def train_network(dataloader):
    network = Net(num_class=10, pretrained_path='results/128_0.5_200_128_500_model_best_loss.pth')
    model_save_dir = './models_stage_simclr'
    model_save_path = os.path.join(model_save_dir, 'train2')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        os.makedirs(os.path.join(model_save_path, 'class_activation_train'))
        os.makedirs(os.path.join(model_save_path, 'class_activation_test'))
        os.makedirs(os.path.join(model_save_path, 'snapshots'))

    snapshot_path = os.path.join(model_save_path, 'snapshots')
    filter_save_path = os.path.join(model_save_path, 'class_activation_{}'.format(args.split))

    for ix, ch in enumerate(network.f.named_children()):
        if ch[0] == args.layer:
            print(ch)
            ch[1].register_forward_hook(get_activation(ch[0]))

    if args.layer == '5':
        filter_size = 1024
    elif args.layer == '6':
        filter_size = 2048
    elif args.layer == '7':
        filter_size = 2048

    
    class_wise_activations = {i: [] for i in range(len(label_names))}
    for didx, (Xs, Ys) in enumerate(dataloader):
        Xs, Ys = Xs.numpy(), Ys.numpy()
        test_function(Xs, Ys, network)

        act_mean = nn.functional.avg_pool2d(activation[args.layer], kernel_size=activation[args.layer].size(2)).squeeze()
        class_wise_activations[Ys[0]].append(act_mean.cpu().numpy())

    plot_mean_activations(filter_save_path, class_wise_activations)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    # -- Assign GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # -- Assertions
    assert (args.dataset)

    # -- Setting seeds for reproducability
    np.random.seed(11)
    random.seed(11)
    torch.manual_seed(11)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(11)
    torch.cuda.manual_seed_all(11)

    # -- Dataset paths
    if args.dataset == "cifar10":
        path = '../../dataset/'
        trainloader, validloader = get_train_valid_loader(path, args.batch_size // 4, augment=False, random_seed=args.seed)
        testloader = get_test_loader(path, batch_size=args.batch_size // 4, shuffle=False, num_workers=2)


    model_save_dir = './models'

    batch_size = args.batch_size

    # -- Train the model
    if args.split == 'train':
        train_network(trainloader)
    else:
        train_network(testloader)
