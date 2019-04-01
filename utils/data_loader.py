import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.timeseries import TimeSeries


def load_daily_data(path, tickers):
	df = pd.read_csv(path)
	df.columns = ['PermNo', 'Date', 'Ticker', 'Low', 'High', 'Close', 'Volume', 'Open']
	df['Date'] = pd.to_datetime(df.Date, format='%m/%d/%Y')
	df = df.sort_values(['Date'])
	dfs = [df[df.Ticker == ticker] for ticker in tickers]
	
	return dfs
	
def create_split(df, training_window=5, prediction_window=3):
	data = df['High'].values
	training_size = math.floor(0.9 * len(data))
	test_size = len(data) - training_size
	train_data = data[0:training_size]
	test_data = data[training_size:]

	train_data = train_data.reshape(-1, 1)
	test_data = test_data.reshape(-1, 1)
	scaler = MinMaxScaler()
	scaler.fit(train_data)

	train_data = scaler.transform(train_data)
	test_data = scaler.transform(test_data)

	train_data = train_data.reshape(-1)
	test_data = test_data.reshape(-1)

	x_train, y_train, x_val, y_val = [], [], [], []

	for i in range(0, training_size - prediction_window - training_window + 1):
		x_train.append(train_data[i:i+training_window])
		y_train.append(train_data[i+training_window:i+training_window+prediction_window])

	for i in range(0, test_size - prediction_window - training_window + 1):
		x_val.append(test_data[i:i+training_window])
		y_val.append(test_data[i+training_window:i+training_window+prediction_window])
		
	return np.asarray(x_train), np.asarray(y_train), np.asarray(x_val), np.asarray(y_val)
	
	

class BatchIterator(object):
    def __init__(self, data, batch_size):
        self.x = data[0]
        self.y = data[1]
        self.batch_size = batch_size
        self.low = 0
        self.high = batch_size
    
    def __getitem__(self, i):
        return (self.x[i], self.y[i])

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        return self

    def __next__(self):
        if self.low >= len(self):
            raise StopIteration
        else:
            batch_x = self.x[self.low : self.high] if self.high < len(self.x) else self.x[self.low:]
            batch_y = self.y[self.low : self.high] if self.high < len(self.y) else self.y[self.low:]
            self.low += self.batch_size
            self.high += self.batch_size
            return batch_x, batch_y
    
    def reset(self):
        self.low = 0
        self.high = self.batch_size