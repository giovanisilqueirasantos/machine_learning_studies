import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from datasets.WineDataset import WineDataset
from utils.graph import plot_decision_regions

dataset = WineDataset(test_size=0.3, random_state=0)
pca = PCA(n_components=2)
lr = LogisticRegression()
x_pca_train = pca.fit_transform(dataset.X_std_train)
x_pca_test = pca.transform(dataset.X_std_test)
lr.fit(x_pca_train, dataset.y_train)

plot_decision_regions(X=x_pca_train, y=dataset.y_train, classifier=lr)
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.legend(loc='lower left')
plt.show()