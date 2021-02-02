# stock-predictor
This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM - multivariate)<br/>to predict the opening stock price of 'Apple' for 30 days using the past 60-day stock price.

### Getting the data

The stock data was gathered from yahoo finance using pandas_datareader package.\
Timeframe is from the 1st of January 2016 to the 1st of January 2021.

```df = web.DataReader('AAPL', data_source='yahoo', start='2016-01-01', end='2021-01-01')```
