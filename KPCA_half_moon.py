from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from utils.rbfKernelPCA import rbf_kernel_pca

def plot_data(X_pca, title):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    fig.suptitle(title)
    ax[0].scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_pca[y==0, 0], np.zeros((50, 1))+0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_pca[y==1, 0], np.zeros((50, 1))-0.02, color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    plt.show()

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row  in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas/ lambdas)

X, y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.show()

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
plot_data(X_spca, 'With out kernel trick')

X_kpca, _ = rbf_kernel_pca(X, gamma=15, n_components=2)
plot_data(X_kpca, 'With kernel trick')

alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)
x_new = X[25]
x_proj = alphas[25]
x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
plt.scatter(alphas[y==0, 0], np.zeros((50)), color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y==1, 0], np.zeros((50)), color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black', label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_reproj, 0, color='green', label='remapped point X[25]', marker='x', s=500)
plt.legend(scatterpoints=1)
plt.show()