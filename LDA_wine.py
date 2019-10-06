import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datasets.WineDataset import WineDataset
from utils.graph import plot_decision_regions

# number of features 
DIMENSION = 13

dataset = WineDataset(test_size=0.3, random_state=0)
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(dataset.X_std_train[dataset.y_train == label], axis=0))
print(mean_vecs)

S_W = np.zeros((DIMENSION, DIMENSION))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter = np.cov(dataset.X_std_train[dataset.y_train == label].T)
    S_W += class_scatter
print(S_W.shape)

mean_overall = np.mean(dataset.X_std_train, axis=0).reshape(DIMENSION, 1)
S_B = np.zeros((DIMENSION, DIMENSION))
for i, mean_vec in enumerate(mean_vecs):
    n = dataset.X_std_train[dataset.y_train == i+1, :].shape[0]
    mean_vec = mean_vec.reshape(DIMENSION, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print(S_B.shape)

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# the first two linear discriminats (eigenvectors) capture the most class-discrimitatory information
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid', label='cumulative "discriminability')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.show()

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
X_lda_train = dataset.X_std_train.dot(w)

lr = LogisticRegression()
lr.fit(X_lda_train, dataset.y_train)
plot_decision_regions(X=X_lda_train, y=dataset.y_train, classifier=lr)
plt.xlabel('ld1')
plt.ylabel('ld2')
plt.legend(loc='lower left')
plt.show()