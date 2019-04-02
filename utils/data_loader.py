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
    highs = df['High'].values
    volumes = df['Volume'].values
    scaler = MinMaxScaler()
    
    # Create training and testing data
    training_size = math.floor(0.9 * len(highs))
    test_size = len(highs) - training_size
    
    train_highs = highs[0:training_size]
    train_volumes = volumes[0:training_size] 
    test_highs = highs[training_size:]
    test_volumes = volumes[training_size:]

    train_highs = train_highs.reshape(-1, 1)
    train_volumes = train_volumes.reshape(-1, 1)
    test_highs = test_highs.reshape(-1, 1)
    test_volumes = test_volumes.reshape(-1, 1)
       
    scaler.fit(train_highs)
    train_highs = scaler.transform(train_highs)
    test_highs = scaler.transform(test_highs)
    train_highs = train_highs.reshape(-1)
    test_highs = test_highs.reshape(-1)
    
    scaler.fit(train_volumes)
    train_volumes = scaler.transform(train_volumes)
    test_volumes = scaler.transform(test_volumes)
    train_volumes = train_volumes.reshape(-1)
    test_volumes = test_volumes.reshape(-1)
    
    train_data = [(train_highs[i], train_volumes[i]) for i in range(training_size)]
    test_data = [(test_highs[i], test_volumes[i]) for i in range(test_size)]
    
    x_train, y_train, x_val, y_val = [], [], [], []

    for i in range(0, training_size - prediction_window - training_window + 1):
        x_train.append(train_data[i:i+training_window]) # Change to train_highs to use only the highs as features (baseline)
        y_train.append(train_highs[i+training_window:i+training_window+prediction_window])

    for i in range(0, test_size - prediction_window - training_window + 1):
        x_val.append(test_data[i:i+training_window]) # Change to test_highs to only use the highs as features (baseline)
        y_val.append(test_highs[i+training_window:i+training_window+prediction_window])

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