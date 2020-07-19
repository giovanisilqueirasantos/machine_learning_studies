import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = [f'ID_{i}' for i in range(20)]
X = np.random.random_sample([20, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:,0], X[:,1], X[:,2], 'gray')
plt.show()

row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
row_clusters = linkage(row_dist, method='complete', metric='euclidean')

df_clusters = pd.DataFrame(row_clusters, columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'], index=[f'cluster {i+1}' for i in range(row_clusters.shape[0])])

row_dendr = dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()