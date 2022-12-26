import pandas as pd

result = pd.read_csv("csv/country_wise_latest.csv")
def read_csv():
    return result
def csv_position():
    csvPos = result.iloc[:, [8, 9]].values
    return csvPos

