import matplotlib.pyplot as plt
import readcsv
from sklearn.cluster import KMeans

x = readcsv.csv_position()

kmeans = KMeans(n_clusters = 2, init='k-means++', n_init=10, max_iter=300, random_state = 0)
y = kmeans.fit_predict(x)

plt.scatter(x[y == 0, 0], x[y == 0, 1], s = 30, c = 'red', label = 'Cluster 1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], s = 30, c = 'blue', label = 'Cluster 2')
# plt.scatter(x[y == 2, 0], x[y == 2, 1], s = 30, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 70, c = 'cyan', label = 'Centroids')
plt.title('Covid-19 Kmeans')
plt.xlabel('-')
plt.ylabel('-')
plt.legend()
plt.show()