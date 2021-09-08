import random
import numpy as np
import pickle
import scipy.special
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import normalize

from sinkhorn import SinkhornSolver
from sinkhorn_utils import *

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

def read_activations():
    filter_save_path = 'models_stage_simclr/train2/class_activation_{}/'
    layer = '6'
    class_wise_activations_test = pickle.load(open(filter_save_path.format('test') + layer + '_activations.pkl','rb'))
    class_wise_activations_train = pickle.load(open(filter_save_path.format('train') + layer + '_activations.pkl','rb'))

    cls = range(len(label_names))
    X_train = np.concatenate(tuple([np.array(class_wise_activations_train[cl][:4000]) for cl in cls]))
    X_train = X_train.reshape(X_train.shape[0], -1)


    X_test = np.concatenate(tuple([np.array(class_wise_activations_test[cl]) for cl in cls]))
    X_test = X_test.reshape(X_test.shape[0], -1)

    X = np.concatenate([X_train, X_test])
    X = normalize(X)

    X_train = X[:X_train.shape[0]]
    X_test = X[X_train.shape[0]:]

    GT_train = np.concatenate(tuple([cl * np.ones(len(class_wise_activations_train[cl][:4000])) for cl in cls]))
    GT_test = np.concatenate(tuple([cl * np.ones(len(class_wise_activations_test[cl])) for cl in cls]))

    return X_train, X_test, None, GT_train, GT_test

def sampleWvec(length):
    ranks = np.array([[6,5,3,1,4,0,2,7,8]
                ,[7,1,3,2,4,0,5,6,8]
                ,[4,0,6,8,5,7,3,2,1]
                ,[2,1,7,4,8,5,6,0,3]
                ,[2,1,8,4,6,5,7,0,3]
                ,[2,1,6,8,5,4,7,0,3]
                ,[3,0,8,6,7,5,4,1,2]
                ,[2,0,5,6,7,8,4,1,3]
                ,[8,4,5,2,0,3,1,6,7]
                ,[7,8,1,2,3,5,0,6,4]])

    vecs = np.array([0.34306195,  0.06861239, -0.11435398, -0.15247198,
           -0.19058997, -0.22870797, -0.26682596, -0.30494396, -0.34306195])

    Y_ = []
    for i in np.random.randint(0, 10, length):
        r = ranks[i].copy()
        r2 = r.copy()
        np.random.shuffle(r)
        r2[r2 < 3] = r[r < 3]
        yt = np.sort(vecs)[r2]
        yt = np.insert(yt, i, 0.686)
        Y_.append(yt)

    Y_train = np.array(Y_)
    return Y_train


def test_fn(X, Y, fc, batch_size):
    all_gt = []
    all_preds = []
    fc.eval()

    for idx in range(0, X.shape[0] // batch_size):
        x = torch.Tensor(X[batch_size * idx: batch_size * (idx + 1)]).cuda()
        gt = torch.Tensor(Y[batch_size * idx: batch_size * (idx + 1)]).cuda()

        with torch.no_grad():
            preds = torch.argmax(fc(x), dim=1)
        all_gt.extend(gt.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    fc.train()

    all_gt, all_preds = np.array(all_gt), np.array(all_preds)
    acc = float(np.sum(all_gt == all_preds).item()) / len(all_gt)
    return acc

def run_experiment(args):
    batch_size = args.bs
    epsilon = args.sinkhorn_epsilon
    iterations = args.sinkhorn_iterations
    learning_rate = args.lr

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

    if args.debug:
        f = open(os.path.join(args.log_path, 'train_{}.log'.format(seed)), 'w')
        log(f, 'args: ' + str(args))
        log(f, 'DVecs:...')
        
        save_scripts(args.log_path, scripts_to_save=['sinkhorn_main.py', 'utils.py', 'sinkhorn.py'])


    distances_train = np.load("clusters/distances_train.npy")
    distances_test = np.load("clusters/distances_test.npy")
    
    X_train, X_test, _, GT_train, GT_test = read_activations()
    Y_train = sampleWvec(X_train.shape[0])
    GT_cluster = np.argmin(distances_test, axis=1)

    distances_train_ = np.zeros((len(GT_train), 10))
    distances_train_[range(len(GT_train)), np.argmin(distances_train, axis=1)] = 1.
    distances_train = distances_train_
    assert(np.all(distances_train >= -1.))
    assert(np.all(distances_train <= 1.))

    centroids = torch.load("clusters/centroids_train.npy")
    centroids = centroids / 10.

    fc = nn.Linear(X_train.shape[1], Y_train.shape[1])
    assert(fc.weight.data.shape == centroids.shape)
    with torch.no_grad():
       fc.weight.data = centroids
    
 
    optimizer = torch.optim.SGD(fc.parameters(), lr=learning_rate,
                                momentum=0.9,
                                weight_decay=5e-4) 

    fc = fc.cuda()

    train_loss = []
    for iteration in range(args.iterations):

        #Randomly sampled batch
        random_indices_x = np.random.randint(0, len(X_train), batch_size)
        x = torch.Tensor(X_train[random_indices_x]).cuda()

        #Randomly sampled prior        
        random_indices = np.random.randint(0, len(X_train), batch_size)
        y = torch.Tensor(Y_train[random_indices]).cuda()
        
        d = torch.Tensor(distances_train[random_indices_x]).cuda()
        d2 = torch.mm(d, d.t())
        
        assert((d2 >= -1.0).all())
        assert((d2 <= 1.0).all()) 
        
        xw = fc(x)

        solver = SinkhornSolver(epsilon=epsilon, iterations=iterations, ground_metric="weighted_l2")
        loss_, pi = solver.forward(xw, y, d2)

        loss = loss_
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(fc.parameters(), 100)
        optimizer.step()
        

        if torch.isnan(loss):
            log(f, "Loss reaching Nan")
            exit()
   
        if iteration % 10 == 0:
            train_loss.append(loss.item())
            if args.debug:
                log(f, "Iter: {}, Loss: {}, Norm: {}, Gradient: {}".format(iteration, loss.item(), fc.weight.data.norm(p=2).item(), fc.weight.grad.norm(p=2)))                

        if args.save_model and iteration % 10 == 0 and np.argmin(train_loss) + 1 == len(train_loss):
            save_checkpoint({
                'epoch': iteration + 1,
                'state_dict': fc.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, args.log_path, "model_seed_{}.pth".format(args.seed))

    if args.debug:
        plot_metrics(args.log_path, args.seed, train_loss)

    fc.load_state_dict(torch.load(os.path.join(args.log_path, "model_seed_{}.pth".format(args.seed)))['state_dict'])

    acc = test_fn(X_test, GT_test, fc, batch_size)
    log(f, "\nTest Accuracy: {}\n".format(acc))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinkhorn Training with Custom Vectors...")
    parser.add_argument('--seed', type=int, default=11, help="seed to select")
    parser.add_argument('--debug', type=bool, default=True, help="turn on debug logs")
    parser.add_argument('--log_path', type=str, default=None, help="location to dump debug logs")
    parser.add_argument('--save_model', type=bool, default=True, help="whether to save the model")

    parser.add_argument('--iterations', type=int, default=10000, help="Number of training iterations")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--bs', type=int, default=500, help="Batch Size")
    parser.add_argument('--sinkhorn_iterations', type=int, default=700, help="Number of iterations in Sinkhorn Assignment")
    parser.add_argument('--sinkhorn_epsilon', type=float, default=0.01, help="Epsilon Regulariser parameter in Sinkhorn Assignment")
    args = parser.parse_args()

    assert(args.log_path is not None)
    acc = run_experiment(args)
