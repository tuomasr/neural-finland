# neural-finland
Predict Finnish electricity spot prices using a neural network trained with publicly available data

## Data
Historical market data can be obtained from http://nordpoolspot.com/historical-market-data/. Note that the prediction accuracy can be improved considerably with proprietary data sources.

## Models
Standard MLP and LSTM networks both deliver relatively good prediction accuracy.

## Dependencies
The model is trained on a GPU by default.
```
numpy
pandas
theano
keras
```
