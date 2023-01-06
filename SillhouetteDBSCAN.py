from sklearn.cluster import DBSCAN
from sklearn import metrics
import readcsv

result = readcsv.read_csv()
x = result.iloc[:, [8, 9]].values

clustering = DBSCAN(eps=4, min_samples=5).fit(x)
labels = clustering.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print("Number of Clusters:",n_clusters_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x, labels))