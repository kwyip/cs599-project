import torch
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim

def train(net, train_itr, val_itr, loss_function, optimizer, device, epochs=10):
	losses = []
	min_val_mse = 200 # Arbitrary value larger than the largest error
	best_model_params = {}
	for epoch in range(epochs):
		i = 0
		train_itr.reset()
		val_itr.reset()
		avg_loss = 0
		val_mse = 0
		net.train()
		for x, y in train_itr:
			net.zero_grad()
			net.hidden_states = net.initialize_hidden_states()
			
			y_pred = net(x)
			y_true = torch.tensor(y, dtype=torch.float, device=device)
			
			loss = loss_function(y_pred, y_true)
			loss.backward()
			optimizer.step()
			
			avg_loss += loss
			i += 1

		print('Epoch:{}, Average loss:{}'.format(epoch, avg_loss/i))
		losses.append(avg_loss)

		num_val = 0
		net.eval()
		for x, y_true in val_itr:
			y_pred = net(x).cpu().detach().numpy()
			val_mse += (np.square(y_true - y_pred)).mean(axis=1).item()
			num_val += 1

		val_mse /= num_val
		print('Average val MSE = {}'.format(val_mse))

		if val_mse < min_val_mse:
			del best_model_params
			min_val_mse = val_mse
			best_model_params = copy.deepcopy(net.state_dict())

	print('************ Finished Training **************')
	print('Minimum val MSE = {}'.format(min_val_mse))
	print('Saving best model..')
	torch.save(best_model_params, './models/best_model.pt')
	
	return losses, min_val_mse

	