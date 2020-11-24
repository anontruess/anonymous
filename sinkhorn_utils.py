from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import shutil
import torch

def save_checkpoint(state, fdir, name='checkpoint.pth'):
    filepath = os.path.join(fdir, name)
    torch.save(state, filepath)

def log(f, txt, do_print=1):
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')

def save_scripts(path, scripts_to_save=None):
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    return

def save_assignments(pi, iteration, split='train'):
    f, axarr = plt.subplots(1, 1, figsize=(18, 9))

    pid = pi.detach()
    #axarr.set_title("Epsilon: {0:}. Cost: {1:.2f}".format(0.1, loss))

    cmap = axarr.imshow(pid)
    axarr.set_title("Probabilistic transport plan")
    cbar = plt.colorbar(cmap, ax=axarr)
    cbar.set_label("Probability mass")
    
    plt.suptitle("Training Assignments")
    plt.savefig('linear/{}_assignments_{}.png'.format(split, iteration))
    plt.close()

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

def save_assignments_test(preds, gt, iteration, split='test'):
    alpha = label_names

    data = confusion_matrix(gt, preds)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest')
    fig.colorbar(cax)

    ax.set_xticks(range(len(alpha)))
    ax.set_yticks(range(len(alpha)))
    ax.set_xticklabels(alpha)
    ax.set_yticklabels(alpha)

    plt.suptitle("Test Assignments")
    plt.savefig('linear/{}_confusion_{}.png'.format(split, iteration))
    plt.close()

def plot_metrics(root_dir, name, train_loss, test_acc=None, cluster_acc=None, test_acc_mapped=None):
    plt.plot(train_loss, color='red', label='train')
    # plt.plot(test_loss, color='black', label='test')
    plt.legend()
    plt.suptitle("Loss")
    plt.ylim(0, 20)
    plt.savefig(os.path.join(root_dir, 'losses_{}.png'.format(name)))
    plt.close()
    
    if test_acc is not None:
        plt.plot(100*np.array(test_acc), label='test')
        plt.legend()
        plt.ylim(0, 100)
        plt.suptitle("Test Accuracy")
        plt.savefig(os.path.join(root_dir, 'accuracies_{}.png'.format(name)))
        plt.close()

    if cluster_acc is not None:
        plt.plot(100*np.array(cluster_acc), label='test')
        plt.legend()
        plt.ylim(0, 100)
        plt.suptitle("Cluster Accuracy")
        plt.savefig(os.path.join(root_dir, 'cluster_accuracies_{}.png'.format(name)))
        plt.close()

    if test_acc_mapped is not None:
        plt.plot(100*np.array(test_acc_mapped), label='test')
        plt.legend()
        plt.ylim(0, 100)
        plt.suptitle("Test Accuracy (mapped)")
        plt.savefig(os.path.join(root_dir, 'mapped_accuracies_{}.png'.format(name)))
        plt.close()

def save_metrics(root_dir, name, unique_preds):
    np.save(os.path.join(root_dir, 'unique_preds_train_{}.npy'.format(name)), unique_preds["train"])
    np.save(os.path.join(root_dir, 'unique_preds_test_{}.npy'.format(name)), unique_preds["test"])

    plt.scatter(range(len(unique_preds["train"])), unique_preds["train"], color='red', label='train')
    plt.scatter(range(len(unique_preds["test"])), unique_preds["test"], color='black', label='test')
    plt.legend()
    plt.suptitle("Unique Predictions")
    plt.ylim(0, 10)
    plt.savefig(os.path.join(root_dir, 'unique_preds_{}.png'.format(name)))
    plt.close()

def Hungarian(A):
    _, col_ind = linear_sum_assignment(A)
    return col_ind

def BestMap(L1, L2):
    L1 = L1.flatten(order='F').astype(float)
    L2 = L2.flatten(order='F').astype(float)
    if L1.size != L2.size:
        sys.exit('size(L1) must == size(L2)')
    Label1 = np.unique(L1)
    nClass1 = Label1.size
    Label2 = np.unique(L2)
    nClass2 = Label2.size
    nClass = max(nClass1, nClass2)

    # For Hungarian - Label2 are Workers, Label1 are Tasks.
    G = np.zeros([nClass, nClass]).astype(float)
    for i in range(0, nClass2):
        for j in range(0, nClass1):
            G[i, j] = np.sum(np.logical_and(L2 == Label2[i], L1 == Label1[j]))

    c = Hungarian(-G)
    print(nClass1, nClass2, c)
    newL2 = np.zeros(L2.shape)
    for i in range(0, nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2, c
