import pandas as pd
import wcss
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('csv/data/CARS_1.csv')

# data['2022_last_updated'] = data['2022_last_updated'].str.replace(',', '').astype(float
data_mapped = data.assign(rank = range(len(data)))
data_mapped = data_mapped.iloc[:,[16,7]]
# cluster = wcss.ElbowMethod(data_mapped)
# print(data)

kmeans = KMeans(2)
identified_clusters = kmeans.fit_predict(data_mapped)
data_with_clusters = data_mapped
data_with_clusters['Cluster'] = identified_clusters
scatter = plt.scatter(data_with_clusters['rank'], data_with_clusters['fuel_tank_capacity'], c=data_with_clusters['Cluster'],
                      cmap='rainbow')
plt.title('Fuel Tank Capacity of Car')
plt.xlabel('Car')
plt.ylabel('Fuel Tank Capacity')
plt.show()



data2 = pd.read_csv('csv/data/Petrol Dataset June 20 2022.csv' ,encoding='latin-1' )
data2_mapped = data2.assign(rank = range(len(data2)))
data2_mapped = data2_mapped.iloc[:,[8,6]]
# cluster2 = wcss.ElbowMethod(data2_mapped)
# print(data2_mapped)

kmeans = KMeans(2)
identified_clusters = kmeans.fit_predict(data2_mapped)
data_with_clusters = data2_mapped
data_with_clusters['Cluster'] = identified_clusters
scatter2 = plt.scatter(data_with_clusters['rank'], data_with_clusters['Price Per Liter (USD)'], c=data_with_clusters['Cluster'],
                      cmap='rainbow')
plt.title('Petrol Price')
plt.xlabel('Country')
plt.ylabel('Petrol Price')
plt.show()


data3 = pd.concat([data,data2], axis=1, join='inner')
data3_mapped = data3.iloc[:,[0,7,22]]

calculator = data3_mapped['Price Per Liter (USD)'].unique()
mean = calculator.mean(axis = 0)

dataAdd = data3_mapped.assign(CalFuel=data3_mapped['fuel_tank_capacity']*mean)
dataAdd = dataAdd.assign(rank = range(len(data3)))
dataAdd_mapped = dataAdd.iloc[:,[4,3]]
# print(dataAdd_mapped)
cluster3 = wcss.ElbowMethod(dataAdd_mapped)

kmeans = KMeans(3)
identified_clusters = kmeans.fit_predict(dataAdd_mapped)
data_with_clusters = dataAdd_mapped
data_with_clusters['Cluster'] = identified_clusters
scatter2 = plt.scatter(data_with_clusters['rank'], data_with_clusters['CalFuel'], c=data_with_clusters['Cluster'],
                      cmap='rainbow')
plt.title('Car Fuel Calculator')
plt.xlabel('Car')
plt.ylabel('Fuel Calculator')
plt.show()
