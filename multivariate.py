# Description: This program uses an artificial recurrent neural network called Long Short Term Memory
#              (LSTM - multivariate) to predict the opening stock price of 'Apple' for 30 days using the past 60 day stock price.

# dataset: https://finance.yahoo.com/quote/AAPL/history/

# imports
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# Business Day US calendar
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# Set the style sheet
plt.style.use('fivethirtyeight')

# Get the stock quote
df = web.DataReader('AAPL', data_source='yahoo', start='2016-01-01', end='2021-01-01')

# Show the data
# print(df)

# Dropping the volume column
df.drop(['Volume'], axis=1, inplace=True)

#Separate dates for future plotting
train_dates = df.index

# Get the number of rows to train the model on
training_data_len = math.ceil( len(df) * .8 )

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)
# print(scaled_data)

# Create the training data set
# Create the scaled training data set
train_data = scaled_data[:training_data_len, :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

n_future = 1
n_past = 60

for i in range(n_past, len(train_data) - n_future + 1):
    x_train.append(train_data[i - n_past:i, 0:df.shape[1]])
    y_train.append(train_data[i + n_future - 1:i + n_future, 2])

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Print the X and Y training sets
print('trainX shape == {}.'.format(x_train.shape))
print('trainY shape == {}.'.format(y_train.shape))

# Define the model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1]))

model.compile(optimizer='adam', loss='mse')
# model.summary()

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

# Plotting the training and validation loss
# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.legend()
# plt.show()

n_future = 30
forecast_period_dates = pd.date_range(list(train_dates)[training_data_len], periods=n_future, freq=us_bd).tolist()

# Create the testing data set
# Create a new array containing scaled test values
test_data = scaled_data[training_data_len - n_past:, :]

# Create the data sets x_test and y_test
x_test = []
y_test = df[training_data_len:]

for i in range(n_past, len(test_data) - n_future + 1):
    x_test.append(test_data[i - n_past:i, 0:df.shape[1]])

# Convert the data to numpy array
x_test = np.array(x_test)

# Get the models predicted price values
predictions = model.predict(x_test[:n_future])

#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform
predictions = np.repeat(predictions, df.shape[1], axis=-1)
predictions = scaler.inverse_transform(predictions)[:,0]

# Convert timestamp to date
forecast_dates = []
for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())

# Convert to numpy arrays
predictions = np.array(predictions)
forecast_dates = np.array(forecast_dates)

# Match the predicted open price with the correspoding dates
df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':predictions})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])
df_forecast.set_index('Date', inplace=True)

# Match the original open price with the correspoding dates
original = df[forecast_dates[0]:forecast_dates[-1]]
original = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':original['Open']})
original['Date']=pd.to_datetime(original['Date'])
original.set_index('Date', inplace=True)

# Plot the original and the predicted open price
original['Open'].plot(label='Real prices', figsize=(16,8))
df_forecast['Open'].plot(label='Predicted')
plt.title('Predicted vs Real Open price for Apple')
plt.xlabel('Date')
plt.ylabel('Open Price USD ($)')
plt.legend()
plt.show()