import readcsv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

result = readcsv.read_csv()
x = readcsv.csv_position()

ss = []
for i in range(2,len(x)):
    km = KMeans(n_clusters = i)
    km.fit(x)
    sil_avg = silhouette_score(x, km.labels_).round(4)
    ss.append([sil_avg, i])

print(ss)
