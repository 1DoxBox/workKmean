import matplotlib.pyplot as plt
import readcsv
from sklearn.cluster import DBSCAN

result = readcsv.csv_position()

dbscan = DBSCAN(eps=4, min_samples=5)
labels = dbscan.fit_predict(result)

plt.scatter(result[labels == -1,0],result[labels == -1,1], s=30, c='blue')
plt.scatter(result[labels == 0,0],result[labels == 0,1], s=30, c='red')
plt.scatter(result[labels == 1,0],result[labels == 1,1], s=30, c='green')
plt.title('Covid-19 DBSCAN')
plt.show()