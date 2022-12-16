import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import readcsv
from sklearn.cluster import KMeans

result = readcsv.read_csv()

result_clustering = result.drop(columns = ['team'])
x = result_clustering.iloc[:, [7,9,10]].values

# wcss = []
for i in range(1, 32):
    kmeans = KMeans(n_clusters = 3, init='k-means++', random_state = 0)
    kmeans.fit(x)
    y = kmeans.fit_predict(x)


plt.scatter(x[y == 0, 0], x[y == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y == 2, 0], x[y == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'cyan', label = 'Centroids')
plt.title('Football World Cup Goals')
plt.xlabel('All Goals')
plt.ylabel('Goals Different')
plt.legend()
plt.show()