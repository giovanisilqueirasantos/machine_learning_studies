from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from datasets.WineDataset import WineDataset

dataset = WineDataset(test_size=0.3, random_state=0)
feat_labels = dataset.df.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(dataset.X_train, dataset.y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

plt.title('Feature Importances')
plt.bar(range(dataset.X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(dataset.X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, dataset.X_train.shape[1]])
plt.tight_layout()
plt.show()