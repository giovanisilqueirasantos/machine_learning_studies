import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from datasets.WineDataset import WineDataset
from utils.graph import plot_decision_regions

dataset = WineDataset(test_size=0.3, random_state=0)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda_train = lda.fit_transform(dataset.X_std_train, dataset.y_train)

lr = LogisticRegression()
lr = lr.fit(X_lda_train, dataset.y_train)
plot_decision_regions(X=X_lda_train, y=dataset.y_train, classifier=lr)
plt.xlabel('ld1')
plt.ylabel('ld2')
plt.legend(loc='lower left')
plt.show()