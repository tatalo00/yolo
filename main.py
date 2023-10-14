import numpy as np
import pandas as pd
import sklearn
from pandas import to_datetime
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
from math import sqrt



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


def make_training_set(data, city, pollen, output_csv):
    pollen = pd.read_csv(pollen)
    data_city = data[city]
    data_city.drop("Unnamed: 0", axis="columns", inplace = True)
    #data_city.to_csv("Data_Beograd.csv", index=False)
    #pollen_train = pd.read_csv("pollen_train.csv")
    pollen_city = pollen[pollen.location == city]
    pollen_city = pollen_city[['date', 'AMBROSIA']]
    #pollen_city.to_csv("Polen_Beograd.csv", index=False)
    pollen_city = pd.DataFrame(pollen_city)
    temp = pd.merge(data_city, pollen_city, on='date', how='left')
    temp['AMBROSIA'].fillna(0, inplace=True)
    temp.to_csv(output_csv, index=False)

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
    #print(key)
    all_data[key] = preprocess_data(value)


for key, value in all_data.items():
    print(key)
    all_data[key] = preprocess_data(value)


#training_set = pd.read_csv('pollen_train.csv')
#training_set = training_set[['location', 'date', 'AMBROSIA']]


make_training_set(all_data, 'БЕОГРАД - НОВИ БЕОГРАД', 'pollen_train.csv', 'train_bg.csv')

#NEURALNA MREZA LSTM
"""
quantitative = ['tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'wpgt', 'pres']
lstm_data = data[quantitative]
window_size = 10  # Number of past days to consider
forecast_horizon = 3  # Number of future days to predict

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(all_data['БЕОГРАД - НОВИ БЕОГРАД'][quantitative])

X, y = [], []

for i in range(len(scaled_data) - window_size - forecast_horizon + 1):
    X.append(scaled_data[i:i+window_size])
    y.append(scaled_data[i+window_size:i+window_size+forecast_horizon])

X = np.array(X)
y = np.array(y)
X_test = pd.read_csv('pollen_test.csv')
y_test = pd.read_csv('pollen_test.csv')
y_test = y_test.AMBROSIA

# Build the LSTM model
model = Sequential()

model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(window_size, X.shape[2])))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu'))
model.add(Dense(forecast_horizon))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Inverse transform the scaled data to get actual values
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Calculate RMSE to evaluate the model's performance
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot the actual and predicted values
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted', color='red')
plt.xlabel('Time')
plt.ylabel('Ragweed Concentration')
plt.title('Ragweed Concentration Prediction')
plt.legend()
plt.show()
"""

