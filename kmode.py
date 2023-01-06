import pandas as pd
import pandas as ps
import numpy as np
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

# %matplotlib inline

data = pd.read_csv("csv/data/CARS_1.csv")
poscsv = data.iloc[:,[0,7]]

cost = []
K = range(1, 5)
for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init="random", n_init=5, verbose=1)
    kmode.fit_predict(poscsv)
    cost.append(kmode.cost_)

plt.plot(K, cost, 'bx-')
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show()

# kmode = KModes(n_clusters=2, init="random", n_init = 5, verbose=1)
# clusters = kmode.fit_predict(poscsv)
# print(clusters)

# data.insert(0, "Cluster", clusters, True)
# print(poscsv)