import pickle
from scipy import spatial
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

D_OUT = 3 # number of dimensions in the encoded vectors
H1_DIM = 64 # dimension of first hidden layer
H2_DIM = 12 # dimension of second hidden layer


class SimMatrixEmbeddingLoss(nn.Module):
	def __init__(self, sim_matrix):
		super(SimMatrixEmbeddingLoss, self).__init__()
		self.sim_matrix = sim_matrix


	def cross_difference(self, input1, input2):
		'''
		input1 - 2D tensor of dimension U x P
		input2 - 2D tensor of dimension V x P
		Computes the cosine similarity of each pairwise combination of users and word 1D tensors
		Returns a resulting matrix of dimension V x W
		'''
		u = input1.size()[0]
		v = input2.size()[0]
		res = Variable(torch.Tensor(u, v))
		for i in range(u):
			for j in range(v):
				vec1 = torch.unsqueeze(input1[i,:], 0)
				vec2 = torch.unsqueeze(input2[j,:], 0)
				res[i][j] = F.cosine_similarity(vec1, vec2)
		return res


	def forward(self, input1, input2):
		'''
		Takes users and words, computes cosine similarity between them and compares them to
		word-user similarity matrix to compute the loss
		'''
		cross_diffs = self.cross_difference(input1, input2)
		return torch.sum((int(self.sim_matrix) - cross_diffs)**2)


class DeepAutoencoder(nn.Module):
	def __init__(self, D_in, H1, H2, D_out):
		super(DeepAutoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(D_in, H1),
			nn.ReLU(True),
			nn.Linear(H1, H2),
			nn.ReLU(True),
			nn.Linear(H2, D_out),
			nn.ReLU(True)
			)
		self.decoder = nn.Sequential(
			nn.Linear(D_out, H2),
			nn.ReLU(True),
			nn.Linear(H2, H1),
			nn.ReLU(True),
			nn.Linear(H1, D_in),
			nn.ReLU(),
			)


	def forward(self, x):
		y = self.encoder(x)
		#x_pred = self.decoder(x)
		return y

#initialize the data
print("Loading similarity matrix data")
with open('../data/autoencoder_sim_matrices.pkl', 'rb') as matrix_f:
	matrix_data = pickle.load(matrix_f)

user_word_matrix = matrix_data['user_word_matrix']
friend_matrix = matrix_data['friend_matrix']
user_num = friend_matrix.shape[0]
word_num = user_word_matrix.shape[1]

# random Tensors to hold inputs
print("Initializing input vectors")
X_user = Variable(torch.sparse.torch.eye(user_num))
X_word = Variable(torch.sparse.torch.eye(word_num))


# We will keep track of the loss at each iteration
losses = []

# Initialize the model and the optimizer
print("Initializing models and loss functions")
model_user = DeepAutoencoder(user_num, H1_DIM, H2_DIM, D_OUT)
model_word = DeepAutoencoder(word_num, H1_DIM, H2_DIM, D_OUT)
model_parameters = list(model_word.parameters()) +  list(model_user.parameters())

#Adam outperforms stochastic gradient descent
optimizer = optim.Adam(model_parameters, lr=0.001)

# Construct our loss functions:
friend_criterion = SimMatrixEmbeddingLoss(friend_matrix)
word_criterion = SimMatrixEmbeddingLoss(user_word_matrix)

print("Begin training...")
for epoch in range(2):
	total_loss = 0
	y_user_pred = model_user(X_user)
	y_word_pred = model_word(X_word)

	optimizer.zero_grad()

	friend_loss = friend_criterion(y_user_pred, y_user_pred)
	user_word_loss = word_criterion(y_user_pred, y_word_pred)
	loss = friend_loss + user_word_loss

	total_loss += loss

	loss.backward()
	optimizer.step()

	losses.append(loss.data)

print(losses)
