import pandas as pd

result = pd.read_csv("csv/game sales.csv")
def read_csv():
    return result
def csv_position():
    csvPos = result.iloc[:, [2, 6]].values
    return csvPos

