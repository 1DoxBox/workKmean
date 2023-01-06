import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import wcss
import readcsv

data = readcsv.read_csv()
data_mapped = data.copy()

data_mapped['Year'] = data_mapped['Year'].replace(np.nan, 0)
data_mapped.loc[data["Year"] >= 2000, "NA+EU+JP since 2000"] = data_mapped["NA_Sales"] + data_mapped["EU_Sales"] + data_mapped["JP_Sales"]
data_mapped.loc[data["Year"] < 2000, "NA+EU+JP since 2000"] = 0
# data_mapped.loc[data["Year"] < 2000, "NA+EU+JP <2000"] = data_mapped["NA_Sales"] + data_mapped["EU_Sales"] + data_mapped["JP_Sales"]
# data_mapped.loc[data["Year"] >= 2000, "NA+EU+JP <2000"] = 0
data_mapped['NA+EU+JP since 2000'] = data_mapped['NA+EU+JP since 2000'].replace(np.nan, 0)
# data_mapped['NA+EU+JP <2000'] = data_mapped['NA+EU+JP <2000'].replace(np.nan, 0)
data_mapped['Platform'] = data_mapped['Platform'].map({'Wii':1, 'NES':2, 'GB':3, 'DS':4, 'X360':5, 'PS3':6, 'PS2':7, 'SNES':8, 'GBA':9, '3DS':10, 'PS4':11, 'N64':12, 'PS':13, 'XB':14, 'PC':15, '2600':16, 'PSP':17, 'XOne':18, 'GC':19, 'WiiU':20, 'GEN':21, 'DC':22, 'PSV':23, 'SAT':24,
 'SCD':25, 'WS':26, 'NG':27, 'TG16':28, '3DO':29, 'GG':30, 'PCFX':31})

print(data_mapped.iloc[:,[1,3,11]])
dataPos = data_mapped.iloc[:,[2,11]]

# print(dataPos.loc[dataPos['Rank'] == 5627])

x = wcss.ElbowMethod(dataPos)

kmeans = KMeans(2)
identified_clusters = kmeans.fit_predict(dataPos)
data_with_clusters = data_mapped.copy()
data_with_clusters['Cluster'] = identified_clusters
scatter = plt.scatter(data_with_clusters['Platform'], data_with_clusters['NA+EU+JP since 2000'], c=data_with_clusters['Cluster'],
                      cmap='rainbow')
plt.title('Game Sales since 2000')
plt.xlabel('Platform')
plt.ylabel('NA+EU+JP SALES')
plt.show()