# stock-price-forecast
This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM - multivariate)<br/>to predict the opening stock price of 'Apple' for 30 days using the past 60-day stock price.

## data

The stock data was gathered from yahoo finance using the pandas_datareader package. <br>
The stock used in this example is 'Apple' (AAPL). <br>
The timeframe is from the 1st of January 2016 to the 1st of January 2021.

Example - first 5 rows

|| High | Low | Open | Close | Adj Close |
|:-------------:|:-------------:|:-------------:|:-------------:|:--------------|:--------------|
|2016-01-04|26.342501|25.500000|25.652500|26.337500|24.400942|
|2016-01-05|26.462500|25.602501|26.437500|25.677500|23.789471|
|2016-01-06|25.592501|24.967501|25.139999|25.174999|23.323915|
|2016-01-07|25.032499|24.107500|24.670000|24.112499|22.339539|
|2016-01-08|24.777500|24.190001|24.637501|24.240000|22.457672|

## model

#### prediction plot

![Plot of the predicted vs actual prices](https://github.com/breezy11/stock-predictor/blob/master/plots/predicted.png)

#### loss

![Plot of the training and validation loss](https://github.com/breezy11/stock-predictor/blob/master/plots/training-validation-loss.png)

