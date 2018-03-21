from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

def read_edge_array(path_to_edgelist):
	'''
	Read a file containing an edgelist into a list.
	@param path_to_edgelist 	The path to the edgelist

	@return result		A list of tuples (source, destination)
	'''

	result = []

	with open(path_to_edgelist) as fp:
		for line in fp:
			first, second = [int(x) for x in line.split()]
			e = (first, second)
			result.append(e)

	return result


def generate_similarity_matrix(path_to_embeddings):
	'''
	Generate the similarity matrix from the embedded node vectors.
	The similarities between two vectors will be computed using the cosine similarity
	@param path_to_embeddings	The location of the file which contains the embeddings of the nodes
								The first line should contain two integers, the number of nodes and the number of dimensions in the embedded space
								Each of the next lines contain the index of the node followed by that node's embedded vector representation

	@return S 	The similarity matrix.
	'''
	first = True #flag to keep track of whether or not we are looking at the first line of the file
	num_nodes = 0 #number of nodes in the graph
	num_dimensions = 0 #number of dimensions in the embedded space
	node_embeddings_matrix = None #the embedding similarity matrix, not initialized until we read the dimensions
	with open(path_to_embeddings) as fp:
		#read the number of nodes and dimension from the first line of the file
		num_nodes, num_dimensions = [int(x) for x in next(fp).split()]

		#we need to store the node embeddings in order in the matrix
		size = (num_nodes, num_dimensions)
		node_embeddings_matrix = np.empty((size))

		#update the rows in the embeddings matrix with the appropriate values
		for line in fp:
			nums = [int(x) for x in line.split()]
			node_index = nums[0]
			embedded_vector = nums[1:]
			node_embeddings_matrix[node_index] = embedded_vector

	#now time to initialize similarity matrix S to a matrix of the appropriate size, filled with zeros
	size = (num_nodes, num_nodes)
	S = np.zeros(size)

	for i in range(0, num_nodes):
		#get the values returned combining the current row with each of the other rows in the matrix
		cos_values_array = cosine_similarity(node_embeddings_matrix.getrow(i), node_embeddings_matrix)
		#store these values as the current row in the similarity matrix
		S[i] = cos_values_array

	return S

def get_edge_scores(S, E_test, E_fake):
	'''
	Read the score in the similarity matrix for each of the edges being used in testing.
	@param S 		The similarity matrix
	@param E_test	List of actual edges that were removed prior to generating the embedding.
	@param E_fake	List of edges that never existed in the graph.

	@return y_scores 	A list containing the scores for the removed edges followed by the scores for the nonexistent edges
	'''

	y_scores = []
	for i in range(len(E_test)):
		e = E_test[i]
		node1 = e[0]
		node2 = e[1]
		score = S[node1][node2]
		y_scores.append(score)

	for i in range(len(E_fake)):
		e = E_fake[i]
		node1 = e[0]
		node2 = e[1]
		score = S[node1][node2]
		y_scores.append(score)

	return y_scores


def main():
	#step 1 (completed before this file is reached): generate edge lists
	E_test = read_edge_array('edgelists/testing.edgelist')
	E_fake = read_edge_array('edgelists/fake.edgelist')
	#step 2 (completed before this file is reached): embed nodes based on E_train

	#step 3: create similarity matrix from the embedded node vectors
	S = generate_similarity_matrix('emb/test_embedding.emd')

	#step 4: create arrays to feed into the AUC score computation
	y_true = []
	#Each of the edges in E_test was an actual edge removed, so the true value is 1
	for i in range(len(E_test)):
		y_true.append(1)
	#Each of the edges in E_fake is not a real edge, so the true value is 0
	for i in range(len(E_fake)):
		y_true.append(0)

	y_scores = get_edge_scores(S, E_test, E_fake)

	#step 5: calculate score from y_true and y_scores
	auc_score = roc_auc_score(y_true, y_scores)
	print("The score this embedding recieved for link prediction was " + str(auc_score))


if __name__ == '__main__':
	main()