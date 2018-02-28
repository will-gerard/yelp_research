import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

N = 10 # number of vectors in the data set, i.e. |V|+|W|
D = 3 # number of dimensions in the encoded vectors

class L1EmbeddingLoss(nn.Module):
	def __init__(self, friend_sim_matrix):
		super(L1EmbeddingLoss, self).__init__()
		self.friend_sim_matrix = friend_sim_matrix


	def cross_difference(inputs):
		'''
		Given a list of 1D tensors (2D tensor batch), does the pair wise combination
		of each 1D tensor and finds their difference. Outputs a 2D tensor of differences.
		inputs is a 2D tensor batch where each row is a 1D tensor for the cross difference
		'''
		n = inputs.size()[0]
		res = tensor.zeros(n,n)
		for i in range(n):
			for j in range(n):
				res[i][j] = F.cosine_similarity(inputs[i], inputs[j])
		return res


	def forward(self, inputs):
		'''
		Takes the input and performs a cartesian product with itself to compute
		loss on pairs of the input based on the similarity matrix
		'''
		diffs = cross_difference(inputs)
		return torch.sum((self.friend_sim_matrix - diffs)**2)



class WordEmbeddingLoss(nn.Module):
	def __init__(self, word_sim_matrix):
		super(WordEmbeddingLoss, self).__init__()
		self.word_sim_matrix = word_sim_matrix


	def cross_difference(users, words):
		'''
		users - 2D tensor of dimension V x P
		words - 2D tensor of dimension W x P
		Computes the cosine similarity of each pairwise combination of users and word 1D tensors
		Returns a resulting matrix of dimension V x W
		'''
		v = users.size()[0]
		w = words.size()[0]
		res = np.zeros(v, w)
		for i in range(v):
			for j in range(w):
				res[i][j] = F.cosine_similarity(users[i], words[j])
		return res


	def forward(self, users, words):
		'''
		Takes users and words, computes cosine similarity between them and compares them to
		word-user similarity matrix to compute the loss
		'''
		cross_diffs = cross_difference(users, words)
		return torch.sum((self.word_sim_matrix - cross_diffs)**2)



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
