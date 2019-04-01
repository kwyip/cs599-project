import torch
import numpy as np
import torch.nn as nn

def load_saved_model(path, net):
	net.load_state_dict(torch.load(path))
	net.eval()
	print('Sucessfully loaded model {}'.format(path))
	return net

def evaluate_model(net, test_itr):
	test_itr.reset()
	test_mse = 0
	num_val = 0
	for x, y_true in test_itr:
		y_pred = net(x).cpu().detach().numpy()
		test_mse += (np.square(y_true - y_pred)).mean(axis=1).item()
		num_val += 1

	test_mse /= num_val
	print('Average test MSE = {}'.format(test_mse))