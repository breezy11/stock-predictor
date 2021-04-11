# stock-predictor
This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM - multivariate)<br/>to predict the opening stock price of 'Apple' for 30 days using the past 60-day stock price.

## Preparing the data

#### Getting the data

The stock data was gathered from yahoo finance using pandas_datareader package.\
Timeframe is from the 1st of January 2016 to the 1st of January 2021.

```df = web.DataReader('AAPL', data_source='yahoo', start='2016-01-01', end='2021-01-01')```

#### Shows first 5 rows


|| High | Low | Open | Close | Adj Close |
|:-------------:|:-------------:|:-------------:|:-------------:|:--------------|:--------------|
|2016-01-04|26.342501|25.500000|25.652500|26.337500|24.400942|
|2016-01-05|26.462500|25.602501|26.437500|25.677500|23.789471|
|2016-01-06|25.592501|24.967501|25.139999|25.174999|23.323915|
|2016-01-07|25.032499|24.107500|24.670000|24.112499|22.339539|
|2016-01-08|24.777500|24.190001|24.637501|24.240000|22.457672|


#### Dates separation

Separates the dates for future plotting

```train_dates = df.index```

#### Scaling the data

```
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)
```

#### Creating the training data set

```
x_train = []
y_train = []

n_future = 1
n_past = 60

for i in range(n_past, len(train_data) - n_future + 1):
    x_train.append(train_data[i - n_past:i, 0:df.shape[1]])
    y_train.append(train_data[i + n_future - 1:i + n_future, 2])
```

## Model

#### Defining the model
```
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1]))

model.compile(optimizer='adam', loss='mse')
```

#### Training the model

``` 
history = model.fit(x_train, y_train, epochs=10, batch_size=16, validation_split=0.1, verbose=1)
```

#### Plotting the losses

![Plot of the training and validation loss](https://github.com/breezy11/stock-predictor/blob/master/plots/training-validation-loss.png)

## Test

#### Getting the dates

```
n_future = 30
forecast_period_dates = pd.date_range(list(train_dates)[training_data_len], periods=n_future, freq=us_bd).tolist()
```

#### Preparing the test data

```
test_data = scaled_data[training_data_len - n_past:, :]

x_test = []
y_test = df[training_data_len:]

for i in range(n_past, len(test_data) - n_future + 1):
    x_test.append(test_data[i - n_past:i, 0:df.shape[1]])

x_test = np.array(x_test)
```

#### Predicting the values

```
predictions = model.predict(x_test[:n_future])
```

#### Inverse transformation

Perform inverse transformation to rescale back to original range
```
predictions = np.repeat(predictions, df.shape[1], axis=-1)
predictions = scaler.inverse_transform(predictions)[:,0]
```

#### Matching the data and dates

Match the predicted open price with the correspoding dates
```
df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':predictions})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])
df_forecast.set_index('Date', inplace=True)
```
Match the original open price with the correspoding dates
```
original = df[forecast_dates[0]:forecast_dates[-1]]
original = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':original['Open']})
original['Date']=pd.to_datetime(original['Date'])
original.set_index('Date', inplace=True)
```

#### Ploting

```
original['Open'].plot(label='Real prices', figsize=(16,8))
df_forecast['Open'].plot(label='Predicted')
plt.title('Predicted vs Real Open price for Apple')
plt.xlabel('Date')
plt.ylabel('Open Price USD ($)')
plt.legend()
plt.show()
```

![Plot of the predicted vs actual prices](https://github.com/breezy11/stock-predictor/blob/master/plots/predicted.png)
