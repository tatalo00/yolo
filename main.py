import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)


data = pd.read_csv("data/pollen_train.csv")
weather = pd.read_csv("data/weather_data.csv")
weather = pd.DataFrame(weather)
weather['date'] = pd.to_datetime(weather['date'])
data = pd.DataFrame(data)
data['date'] = pd.to_datetime(data['date'])

#plt.figure(1)
#plt.plot(data.date[2000:2100], data.AMBROSIA[2000:2100])
#plt.show()

print(weather.info())
print(weather.describe())

#window_size = 365
#rolling_mean = data['AMBROSIA'].rolling(window=window_size).mean()
#plt.plot(data.date, rolling_mean)
#plt.show()
print("AMBROZIJA\n")

ambrosia_bg = data.AMBROSIA[data['location'] == "БЕОГРАД - НОВИ БЕОГРАД"]
date_x_axis = data['date']
date_x_axis = date_x_axis[data.location == "БЕОГРАД - НОВИ БЕОГРАД"]
#for i in range(len(date_x_axis)):
    #date_x_axis[i] = date_x_axis[i][-5:]
ambrosia_bg = ambrosia_bg.to_numpy()
plt.figure(2)
plt.plot(date_x_axis, ambrosia_bg)
plt.show()
print(ambrosia_bg)

autocorrel = np.correlate(ambrosia_bg, ambrosia_bg)
print(autocorrel)
#plt.figure(3)
#plt.plot(autocorrel)
#plt.show()
