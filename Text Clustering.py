import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('csv/game sales.csv')
data_mapped = data.copy()
# print(data['Platform'].unique())
data_mapped['Platform'] = data_mapped['Platform'].map({'Wii':1, 'NES':2, 'GB':3, 'DS':4, 'X360':5, 'PS3':6, 'PS2':7, 'SNES':8, 'GBA':9, '3DS':10, 'PS4':11, 'N64':12, 'PS':13, 'XB':14, 'PC':15, '2600':16, 'PSP':17, 'XOne':18, 'GC':19, 'WiiU':20, 'GEN':21, 'DC':22, 'PSV':23, 'SAT':24,
 'SCD':25, 'WS':26, 'NG':27, 'TG16':28, '3DO':29, 'GG':30, 'PCFX':31})
# print(data_mapped)

x = data_mapped.iloc[:,[2,6]]
print(x)

kmeans = KMeans(2)
# print(kmeans.fit(x))

identified_clusters = kmeans.fit_predict(x)
# print(identified_clusters)

data_with_clusters = data_mapped.copy()
data_with_clusters['Cluster'] = identified_clusters

scatter = plt.scatter(data_with_clusters['Platform'], data_with_clusters['NA_Sales'], c=data_with_clusters['Cluster'],
                      cmap='rainbow')
# plt.xlim(-180,180)
# plt.ylim(1,100)
plt.title('Game Sales')
plt.xlabel('Platform')
plt.ylabel('NA SALES')
plt.show()