import matplotlib.pyplot as plt
import readcsv
from sklearn.cluster import KMeans
import numpy as np
import numpy.matlib
def ElbowMethod(x):
    wcss = []
    for i in range(1,10):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,10), wcss)
    plt.title('WCSS TEST')
    plt.xlabel('Number of clusters(k)')
    plt.ylabel('WCSS')

    return plt.show()

# result = readcsv.read_csv()
# result['Platform'] = result['Platform'].map({'Wii':1, 'NES':2, 'GB':3, 'DS':4, 'X360':5, 'PS3':6, 'PS2':7, 'SNES':8, 'GBA':9, '3DS':10, 'PS4':11, 'N64':12, 'PS':13, 'XB':14, 'PC':15, '2600':16, 'PSP':17, 'XOne':18, 'GC':19, 'WiiU':20, 'GEN':21, 'DC':22, 'PSV':23, 'SAT':24,
#  'SCD':25, 'WS':26, 'NG':27, 'TG16':28, '3DO':29, 'GG':30, 'PCFX':31})
# x = result.iloc[:, [2,6]]
#
# outp=wcssElbowMethod(x)
# print(outp)


# nPoints = len(wcss)
# allCoord = np.vstack((range(nPoints), wcss)).T
# firstPoint = allCoord[0]
# lineVec = allCoord[-1] - allCoord[0]
# lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
# vecFromFirst = allCoord - firstPoint
# scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
# vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
# vecToLine = vecFromFirst - vecFromFirstParallel
# distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
# idxOfBestPoint = np.argmax(distToLine)
#
# print(f'Optimum number of cluster by Elbow method: {idxOfBestPoint}')