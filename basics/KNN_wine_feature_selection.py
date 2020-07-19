from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from datasets.WineDataset import WineDataset
from utils.SequentialBackwardSelection import SequentialBackwardSelection

knn = KNeighborsClassifier(n_neighbors=2)
sbs = SequentialBackwardSelection(knn, k_features=1)
dataset = WineDataset(test_size=0.3, random_state=0)
sbs.fit(dataset.X_std_train, dataset.y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()