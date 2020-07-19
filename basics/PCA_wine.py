import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datasets.WineDataset import WineDataset
from utils.graph import plot_decision_regions

dataset = WineDataset(test_size=0.3, random_state=0)
cov_mat = np.cov(dataset.X_std_train.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# plot how much the principle components is contributing to variance
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,14), var_exp, alpha=0.5, align='center', label='cumulative explained variance')
plt.step(range(1,14), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

# the trasformation matrix to project 13 dimensional vectors in to 2 dimensional subspace
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))

lr = LogisticRegression()
x_pca_train = dataset.X_std_train.dot(w)
x_pca_test = dataset.X_std_test.dot(w)
lr.fit(x_pca_train, dataset.y_train)

plot_decision_regions(X=x_pca_train, y=dataset.y_train, classifier=lr)
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.legend(loc='lower left')
plt.show()