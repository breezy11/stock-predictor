# stock-predictor
This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM - multivariate)<br/>to predict the opening stock price of 'Apple' for 30 days using the past 60-day stock price.

## Preparing the data

#### Getting the data

The stock data was gathered from yahoo finance using pandas_datareader package.\
Timeframe is from the 1st of January 2016 to the 1st of January 2021.

```df = web.DataReader('AAPL', data_source='yahoo', start='2016-01-01', end='2021-01-01')```

#### Droping the 'Volume' columns since we won't be using it.

```df.drop(['Volume'], axis=1, inplace=True)```

#### Printing the first 5 rows of the dataframe


|| High | Low | Open | Close | Adj Close |
|:-------------:|:-------------:|:-------------:|:-------------:|:--------------|:--------------|
|2016-01-04|26.342501|25.500000|25.652500|26.337500|24.400942|
|2016-01-05|26.462500|25.602501|26.437500|25.677500|23.789471|
|2016-01-06|25.592501|24.967501|25.139999|25.174999|23.323915|
|2016-01-07|25.032499|24.107500|24.670000|24.112499|22.339539|
|2016-01-08|24.777500|24.190001|24.637501|24.240000|22.457672|


#### Separate dates for future plotting

```train_dates = df.index```

#### Scaling the data

```
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)
```

#### Create the training data set

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

#### Define the model
```
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1]))

model.compile(optimizer='adam', loss='mse')
```

#### Train the model

``` 
history = model.fit(x_train, y_train, epochs=10, batch_size=16, validation_split=0.1, verbose=1)
```

#### Plotting the training and validation loss





