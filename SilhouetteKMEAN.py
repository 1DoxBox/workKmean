from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import readcsv

result = readcsv.csv_position()

wcss = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(result)
    wcss.append(kmeans.inertia_)
# print(silhouette_score(result, kmeans.labels_))

kmeans_per_k = [KMeans(n_clusters=k, random_state=0).fit(result) for k in range(1,10)]

silhouette_score = [silhouette_score(result, model.labels_)
                    for model in kmeans_per_k[1:]]
print(silhouette_score)
