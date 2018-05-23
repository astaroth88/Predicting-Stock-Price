#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 23:11:50 2018

@author: astaroth
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# training data
dataset_train = pd.read_csv('Dataset/Google_Stock_Price_Train.csv')
training_data = dataset_train.iloc[:,1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_training_data = scaler.fit_transform(training_data)

# creating a data structure of 60 timestamps and 1 output
X_train = []
y_train = []
for i in range(60, 1257):
    X_train.append(scaled_training_data[i-60:i, 0])
    y_train.append(scaled_training_data[i])
X_train, y_train = np.array(X_train), np.array(y_train)

# reshaping
# reshape size = (no. of samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

# bulding the RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

regressor = Sequential()

# first layer
regressor.add(LSTM(units = 50,
                   return_sequences= True,
                   input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# second layer
regressor.add(LSTM(units = 50,
                   return_sequences= True))
regressor.add(Dropout(0.2))

# third layer
regressor.add(LSTM(units = 50,
                   return_sequences= True))
regressor.add(Dropout(0.2))

# fourth(final) layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(units = 1))

# compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# training
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# testing
dataset_test = pd.read_csv('Dataset/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# predicting the stock price of Jan 2017
dataset_total = pd.concat((dataset_train['Open'],
                           dataset_test['Open']),
                           axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,
                    (X_test.shape[0], X_test.shape[1], 1))
# predicting
predicted_stock_price = regressor.predict(X_test)

# inverse scaling
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price,
         color = 'red',
         label = 'Real Google Stock Price')
plt.plot(predicted_stock_price,
         color = 'blue',
         label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction Jan 2017')
plt.xlabel('Time')
plt.ylabel('Google Stock Price Prediction')
plt.legend()
plt.show()




























































