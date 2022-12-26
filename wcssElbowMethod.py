import matplotlib.pyplot as plt
import readcsv
from sklearn.cluster import KMeans
import numpy as np
import numpy.matlib


result = readcsv.csv_position()

wcss = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(result)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,10), wcss)
plt.title('WCSS TEST')
plt.xlabel('Number of clusters(k)')
plt.ylabel('WCSS')
plt.show()

nPoints = len(wcss)
allCoord = np.vstack((range(nPoints), wcss)).T
firstPoint = allCoord[0]
lineVec = allCoord[-1] - allCoord[0]
lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
vecFromFirst = allCoord - firstPoint
scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
vecToLine = vecFromFirst - vecFromFirstParallel
distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
idxOfBestPoint = np.argmax(distToLine)

print(f'Optimum number of cluster by Elbow method: {idxOfBestPoint}')