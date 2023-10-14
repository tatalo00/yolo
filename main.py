import numpy as np
import pandas as pd
import sklearn
from pandas import to_datetime
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def drop_tavg(data):
    indices = data[data["tavg"].isna()].index
    data = data.drop(indices)
    return data

def drop_tmin(data):
    indices = data[data["tmin"].isna()].index
    data = data.drop(indices)
    return data

def drop_tmax(data):
    indices = data[data["tmax"].isna()].index
    data = data.drop(indices)
    return data

def drop_snow(data):
    data.drop("snow", axis="columns", inplace=True)

def drop_tsun(data):
    data.drop("tsun", axis="columns", inplace=True)

def replace_value_pres(data):
    data.pres.fillna(data.pres.mean(), inplace=True)

def replace_value_wdir(data):
    data.wdir.fillna(data.wdir.mean(), inplace=True)

def replace_value_wspd(data):
    data.wspd.fillna(data.wspd.mean(), inplace=True)

def replace_values_wpgt(data):

    index_train = data[data["wpgt"].notna()].index
    index_test = data[data["wpgt"].isna()].index
    if(len(index_train) == 0):
        data.drop("wpgt", axis="columns", inplace=True)
        return


    test_data = data.loc[index_test]
    train_data = data.loc[index_train]


    train_data_x = train_data[train_data.columns.difference(['prcp', "date", "wpgt"])]
    train_data_y = train_data.loc[:, "wpgt"]
    test_data_x = test_data[test_data.columns.difference(['prcp', "date", "wpgt"])]


    lin_regres = LinearRegression().fit(train_data_x, train_data_y)
    missing_values = lin_regres.predict(test_data_x)

    data.loc[index_test, "wpgt"] = missing_values


def replace_values_prcp(data):
    index_train = data[data["prcp"].notna()].index
    #if(len(index_train) == 0):
        #return
    index_test = data[data["prcp"].isna()].index

    test_data = data.loc[index_test]
    train_data = data.loc[index_train]

    train_data_x = train_data[train_data.columns.difference(['prcp', "date"])]
    train_data_y = train_data.loc[:, "prcp"]
    test_data_x = test_data[test_data.columns.difference(['prcp', "date"])]

    lin_regres = LinearRegression().fit(train_data_x, train_data_y)
    missing_values = lin_regres.predict(test_data_x)

    data.loc[index_test, "prcp"] = missing_values


def preprocess_data(data):

    data = drop_tavg(data)
    data = drop_tmax(data)
    data = drop_tmin(data)
    drop_snow(data)
    drop_tsun(data)
    replace_value_pres(data)
    replace_value_wspd(data)
    replace_value_wdir(data)
    replace_values_wpgt(data)
    replace_values_prcp(data)

    return data

pd.set_option('display.max_columns', None)
data = pd.read_csv("weather_data.csv")

data_beograd = data.loc[data["location"] == 'БЕОГРАД - НОВИ БЕОГРАД']
data_vrsac = data.loc[data["location"] == 'ВРШАЦ']
data_kraljevo = data.loc[data["location"] == 'КРАЉЕВО']
data_kragujevac = data.loc[data["location"] == 'КРАГУЈЕВАЦ']
data_nis = data.loc[data["location"] == 'НИШ']
data_pozarevac = data.loc[data["location"] == 'ПОЖАРЕВАЦ']
data_subotica = data.loc[data["location"] == 'СУБОТИЦА']



data_beograd.drop("location", axis="columns", inplace=True)
data_vrsac.drop("location", axis="columns", inplace=True)
data_kraljevo.drop("location", axis="columns", inplace=True)
data_kragujevac.drop("location", axis="columns", inplace=True)
data_nis.drop("location", axis="columns", inplace=True)
data_pozarevac.drop("location", axis="columns", inplace=True)
data_subotica.drop("location", axis="columns", inplace=True)

all_data = {}
all_data['БЕОГРАД - НОВИ БЕОГРАД'] = data_beograd
all_data['КРАЉЕВО'] = data_kraljevo
all_data['КРАГУЈЕВАЦ'] = data_kragujevac
all_data['НИШ'] = data_nis
all_data['ПОЖАРЕВАЦ'] = data_pozarevac
all_data['СУБОТИЦА'] = data_subotica
all_data['ВРШАЦ'] = data_vrsac

for key, value in all_data.items():
    print(key)
    all_data[key] = preprocess_data(value)

