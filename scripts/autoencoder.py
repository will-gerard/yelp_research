import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

N = 10 # number of vectors in the data set, i.e. |V|+|W|
D = 3 # number of dimensions in the encoded vectors

class DeepAutoencoder(nn.Module):
	def __init__(self):
		super(DeepAutoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(INPUT_VECTOR_SIZE, 64),
			nn.ReLU(True),
			nn.Linear(64, 12),
			nn.ReLU(True),
			nn.Linear(12, 3),
			nn.ReLU(True)
			)
		self.decoder = nn.Sequential(
			nn.Linear(3, 12),
			nn.ReLU(True),
			nn.Linear(12, 64),
			nn.ReLU(True),
			nn.Linear(64, INPUT_VECTOR_SIZE),
			nn.ReLU(),
			)

	def forward(self, x):
		y = self.encoder(x)
		#x_pred = self.decoder(x)

		return y

#input matrix, randomly initialized for now
X = torch.randn(N, N)
#randomly initialize the ouput matrix
Y = torch.randn(N, D)

#we will keep track of the loss at each iteration
losses = []
#initialize the model and the optimizer
#Adam outperforms stochastic gradient descent
model = DeepAutoencoder(nn.Module):
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
	total_loss = 0
	for i in range(N):
		#get a copy of the current row in a Variable object
		x_i = autograd.Variable(X[i].clone())
		y_pred = model(x_i)

		model.zero_grad()

		loss = loss_function(y_pred)

		total_loss += loss

	loss.backward()
	optimizer.step()

	losses.append(loss.data)

print(losses)