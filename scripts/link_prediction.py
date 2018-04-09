'''
After embedding the nodes from the training edge list, performs an AUROC link prediction test
using the missing and removed edge lists.

Takes 4 command line arguments:
1: Path of removed edge list
2: Path of missing edge list
3: Path of embedding

Ex: python3 link_prediction.py ../data/edgelists/user_removed.edgelist ../data/edgelists/user_missing.edgelist ../data/emb/yelp_training_user.emb
'''
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist
from scipy.sparse import lil_matrix
import numpy as np
import sys

REMOVED_PATH = sys.argv[1]
MISSING_PATH = sys.argv[2]
EMBEDDING_PATH = sys.argv[3]

ZERO_INDEXING_NODES = True

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
	with open(path_to_embeddings) as fp:
		#read the number of nodes and dimension from the first line of the file
		num_nodes, num_dimensions = [int(x) for x in next(fp).split()]

		#we need to store the node embeddings in order in the matrix
		size = (num_nodes, num_dimensions)
		#node_embeddings_matrix = np.empty((size))
		node_embeddings_matrix = np.empty(size)

		offset = 0
		if not ZERO_INDEXING_NODES:
			offset = 1

		#update the rows in the embeddings matrix with the appropriate values
		for line in fp:
			nums = [float(x) for x in line.split()]
			node_index = int(nums[0]) - offset #if no node 0 must subtract 1
			embedded_vector = nums[1:]
			node_embeddings_matrix[node_index] = embedded_vector

	#now time to initialize similarity matrix S to a matrix of the appropriate size, filled with zeros
	size = (num_nodes, num_nodes)

	S = 1 - cdist(node_embeddings_matrix, node_embeddings_matrix, 'cosine')
	return S

def get_edge_scores(S, E_test, E_fake):
	'''
	Read the score in the similarity matrix for each of the edges being used in testing.
	@param S 		The similarity matrix
	@param E_test	List of actual edges that were removed prior to generating the embedding.
	@param E_fake	List of edges that never existed in the graph.

	@return y_scores 	A list containing the scores for the removed edges followed by the scores for the nonexistent edges
	'''

	offset = 0
	if not ZERO_INDEXING_NODES:
		offset = 1

	y_scores = []
	for i in range(len(E_test)):
		e = E_test[i]
		node1 = e[0]
		node2 = e[1]
		#need to subtract one if there is no node 0
		score = S[node1 - offset][node2 - offset]
		y_scores.append(score)

	for i in range(len(E_fake)):
		e = E_fake[i]
		node1 = e[0]
		node2 = e[1]
		#need to subtract one if there is no node 0
		score = S[node1 - offset][node2 - offset]
		y_scores.append(score)

	return y_scores


def main():
	#step 1: generate edge lists
	print("READING EDGE LISTS...")
	E_test = read_edge_array(REMOVED_PATH)
	E_fake = read_edge_array(MISSING_PATH)
	#step 2 (completed before this file is reached): embed nodes based on E_train

	#step 3: create similarity matrix from the embedded node vectors
	print("GENERATING SIMILARITY MATRIX...")
	S = generate_similarity_matrix(EMBEDDING_PATH)

	#step 4: create arrays to feed into the AUC score computation
	print("COMPUTING AREA UNDER RECIEVER OPERATING CHARACTERISTIC CURVE...")
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
