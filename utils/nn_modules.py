import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim

class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, prediction_window, bidirectional=False, batch_size=1, p=0):
        super(StockPredictor, self).__init__()
        
        ################### Model Properties ####################
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p = p
        self.batch_size = batch_size
        self.prediction_window = prediction_window
        self.bidirectional = bidirectional
        self.directionality = 2 if bidirectional else 1
        self.device = device
        #########################################################
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size // self.directionality, self.num_layers, dropout=self.p, 
                            bidirectional=self.bidirectional)
        self.hidden_states = self.initialize_hidden_states()
        self.output = nn.Linear(self.hidden_size, self.prediction_window)
        
    def initialize_hidden_states(self):
        return (torch.zeros((self.directionality * self.num_layers, self.batch_size, self.hidden_size // self.directionality), 
                            device=self.device),
                torch.zeros((self.directionality * self.num_layers, self.batch_size, self.hidden_size // self.directionality), 
                            device=self.device))
    
    def forward(self, x):
        # x is of shape torch.size([batch_size, training_window])
        model_in = torch.tensor(x, dtype=torch.float, device=self.device).view(x.shape[1], self.batch_size, self.input_size)
        lstm_out, self.hidden_states = self.lstm(model_in, self.hidden_states)
        
        # Need the output of the last timestep of the LSTM only 
        prediction = self.output(lstm_out[-1].view(self.batch_size, -1))
        
        return prediction