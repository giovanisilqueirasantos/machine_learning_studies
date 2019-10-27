from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from utils.rbfKernelPCA import rbf_kernel_pca

def plot_data(X_pca, title):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    fig.suptitle(title)
    ax[0].scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_pca[y==0, 0], np.zeros((500, 1))+0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_pca[y==1, 0], np.zeros((500, 1))-0.02, color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    plt.show()

X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.show()

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
plot_data(X_spca, 'With out Kernel trick')

X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
plot_data(X_kpca, 'With Kernel trick')