import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import normalize

class SinkhornSolver(nn.Module):
    def __init__(self, epsilon, iterations=100, ground_metric=lambda x: torch.pow(x, 2)):
        super(SinkhornSolver, self).__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self.ground_metric = ground_metric

    def forward(self, x, y):
        num_x = x.size(-2)
        num_y = y.size(-2)
        
        batch_size = 1 if x.dim() == 2 else x.size(0)

        # Marginal densities are empirical measures
        a = x.new_ones((batch_size, num_x), requires_grad=False) / num_x
        b = y.new_ones((batch_size, num_y), requires_grad=False) / num_y
        
        a = a.squeeze()
        b = b.squeeze()
                
        # Initialise approximation vectors in log domain
        u = torch.zeros_like(a)
        v = torch.zeros_like(b)

        # Stopping criterion
        threshold = 1e-1
        
        # Cost matrix
        C = self._compute_cost(x, y)

        # Sinkhorn iterations
        for i in range(self.iterations): 
            u0, v0 = u, v
                        
            # u^{l+1} = a / (K v^l)
            K = self._log_boltzmann_kernel(u, v, C)
            u_ = torch.log(a + 1e-8) - torch.logsumexp(K, dim=1)
            u = self.epsilon * u_ + u
                        
            # v^{l+1} = b / (K^T u^(l+1))
            K_t = self._log_boltzmann_kernel(u, v, C).transpose(-2, -1)
            v_ = torch.log(b + 1e-8) - torch.logsumexp(K_t, dim=1)
            v = self.epsilon * v_ + v
            
            # Size of the change we have performed on u
            diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
            mean_diff = torch.mean(diff)
                        
            if mean_diff.item() < threshold:
                break

        K = self._log_boltzmann_kernel(u, v, C)
        pi = torch.exp(K)
        cost = torch.sum(pi * C, dim=(-2, -1))

        return cost, pi

    def _compute_cost(self, x, y):
        x_ = x.unsqueeze(-2)
        y_ = y.unsqueeze(-3)
        C = torch.sum(self.ground_metric(x_ - y_), dim=-1)
        return C

    def _log_boltzmann_kernel(self, u, v, C=None):
        C = self._compute_cost(x, y) if C is None else C
        kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
        kernel /= self.epsilon
        return kernel

def show_assignments(a, b, P=None, ax=None): 
    if P is not None:
        norm_P = P/P.max()
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                ax.arrow(a[i, 0], a[i, 1], b[j, 0] - a[i, 0], b[j, 1] - a[i, 1],
                        alpha=norm_P[i,j].item(), color="r")

    ax = plt if ax is None else ax
    ax.scatter(*a.t(), color="red")
    ax.scatter(*b.t(), color="blue")

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

def save_assignments(u, y, pi):
    ud = u.detach()
    show_assignments(ud, y)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Input to sinkhorn")
    plt.savefig('clusters/input_{}.png'.format(iteration))

    f, axarr = plt.subplots(2, 2, figsize=(18, 9))

    pid = pi.detach()
    show_assignments(ud, y, pid, axarr[0, 0])
    axarr[0, 0].set_title("Epsilon: {0:}. Cost: {1:.2f}".format(0.01, loss))

    cmap = axarr[1, 0].imshow(pid)
    axarr[1, 0].set_title("Probabilistic transport plan")
    cbar = plt.colorbar(cmap, ax=axarr[1, 0])
    cbar.set_label("Probability mass")

    plt.suptitle("Assignments")
    plt.savefig('clusters/assignments_{}.png'.format(iteration))
    plt.close()

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

    print(X_train.shape)
    print(X_test.shape)
    return X_train, X_test

@torch.no_grad()
def test_fn(X, batch_size, split):
    all_preds = []
    all_distances = []
    n_iterations = X.shape[0] // batch_size
    if X.shape[0] // batch_size == 0:
        n_iterations += 1

    for idx in range(0, n_iterations):
        x = torch.Tensor(X[batch_size * idx: min(batch_size * (idx + 1), X.shape[0])])

        solver = SinkhornSolver(epsilon=0.01, iterations=1000)

        cost = solver._compute_cost(x, u)
        labels = torch.argmin(cost, dim=1)

        all_preds.extend(labels.cpu().numpy())
        all_distances.extend(cost.cpu().numpy())

    all_distances = np.array(all_distances)
    all_preds = np.array(all_preds)

    np.save("clusters/cluster_assignments_{}.npy".format(split), all_preds)
    np.save("clusters/distances_{}.npy".format(split), all_distances)
    torch.save(u, "clusters/centroids_{}.npy".format(split))

if __name__ == "__main__":
    seed = 11
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

    os.mkdir('clusters')

    X_train, X_test = read_activations()

    batch_size = 1000
    dimensions = 2048
    K = 10
    u = torch.randn(K, dimensions, requires_grad=True)
    print(u.size())

    optimizer = torch.optim.Adam([u], lr=0.01)

    min_loss = 10.
    for iteration in range(5000):

        #Randomly sampled batch
        y = torch.Tensor(X_train[np.random.randint(0, len(X_train), batch_size)])

        solver = SinkhornSolver(epsilon=0.01, iterations=1000)
        loss, pi = solver.forward(u, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Loss: ", loss.item())

        # Evaluate
        if iteration % 10 == 0 and min_loss > loss.item():
            min_loss = loss.item()
            test_fn(X_test, batch_size, split='test'.format(loss.item()))                
            test_fn(X_train, batch_size, split='train'.format(loss.item()))
