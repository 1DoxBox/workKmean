import numpy as np
import readcsv


def knn(today):
    result = readcsv.read_csv()
    x = readcsv.csv_position()
    y = result['Country/Region'].values

    P = np.array(today)
    D = np.zeros(len(y))
    for i,dataX in enumerate(x):
        D[i] = np.sqrt(np.sum((P-dataX)**2))

    # minD = np.min(D)
    indexMin = np.argmin(D)
    predictResult = y[indexMin]
    return predictResult

today = [5.7, 88.5]
outp = knn(today)
print(outp)


