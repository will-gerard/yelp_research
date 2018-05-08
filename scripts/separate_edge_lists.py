'''
Separates an edge list into training, missing, and removed lists for AUROC link prediction tests:
Takes in 4 (optional 6) command line parameters:

1: file path to original edge list
2: file path for where to write the training edge list
3: file path for where to write the removed edge list
4: file path for where to write the missing edge list

Optionally, you can restrict the removed and missing edges to only involve nodes and edges up to a certain index
(e.g. missing and removed edges only involve user-user)
We make the assumption that all user nodes are indexed first, and all user-user edges are also indexed first.

5: (optional): max node index that will be involved w/ removal/missing (e.g. 10674)
6: (optional): max edge index that will be involved w/ removal/missing (e.g. 56753)

Ex:
python3 separate_edge_lists.py ../data/friend_edge_list.txt ../data/edgelists/user_training.edgelist ../data/edgelists/user_removed.edgelist ../data/edgelists/user_missing.edgelist
'''
import random
import math
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
import sys

EDGE_LIST_PATH = sys.argv[1]
WRITE_TRAINING_PATH = sys.argv[2]
WRITE_REMOVED_PATH = sys.argv[3]
WRITE_MISSING_PATH = sys.argv[4]
RESTRICT_INDICES = False

if len(sys.argv) > 5:
	RESTRICT_INDICES = True
	MAX_NODE_IDX = int(sys.argv[5])
	MAX_EDGE_IDX = int(sys.argv[6])
else:
	print("Separating on all edges. No index restriction (You should reconsider if you're separating user-word edges)")


def separate_edge_lists(full_edgelist_path, training_percentage=0.9):
	'''
	Turn the full edge list into a list of edges to train on,
	one to test on, and one of nonexistent edges to use in the test.
	@param full_edgelist_path	all edges in the graph in the form source, destination
								with each edge on its own line
	@param training_percentage	the percentage of links to keep during training

	@return tuple (E_train, E_test, E_fake)
	'''
	#read the edges into a set to generate E_fake
	full_edge_list = []
	with open(full_edgelist_path) as fp:
		num_nodes, num_edges = [int(x) for x in next(fp).split()]

		for line in fp:
			first, second = [int(x) for x in line.split()]
			e = (first, second)
			full_edge_list.append(e)
	full_edge_set = set(full_edge_list)

	if RESTRICT_INDICES:
		num_nodes = MAX_NODE_IDX
		num_edges = MAX_EDGE_IDX

	#get the number of elements in the testing set:
	L_train = math.ceil(training_percentage * num_edges)
	L_test = num_edges - L_train

	#generate L random nonexistent links
	E_fake = []
	count = 0
	while count < L_test:
		#pick two random nodes
		nodes = random.sample(range(num_nodes), 2)
		#TODO: clean this up. Is checking both directions necessary?
		node1 = nodes[0]
		node2 = nodes[1]
		edge1 = (node1, node2)
		edge2 = (node2, node1)
		if edge1 not in full_edge_set and edge2 not in full_edge_set:
			#order of nodes shouldn't matter since similarity matrix will be symmetrical
			E_fake.append(edge1)
			count+=1

	#select L random edges to remove
	#get the full list of the edges, we will remove some to create the testing list
	E_train = []
	E_test = []
	#randomly select the edges to remove
	indegrees = calcIndegrees(full_edge_list[:num_edges], num_nodes)
	adj_matrix = lil_matrix(toAdjMatrix(full_edge_list[:num_edges], num_nodes))
	indices_to_remove = list(range(num_edges))
	random.shuffle(indices_to_remove)

	temp_count = 0
	print("Total edges: " + str(len(indices_to_remove)))
	print("We need to remove 10% of those edges")
	for i in indices_to_remove:
		print("Checking edge " + str(temp_count), end='\r')
		edge = full_edge_list[i]
		# try removing the edge
		adj_matrix[edge[0], edge[1]] = 0
		adj_matrix[edge[1], edge[0]] = 0
		N_components, _ = connected_components(adj_matrix, directed=False)

		if indegrees[edge[0]] > 1 and indegrees[edge[1]] > 1 and N_components == 1:
			# Valid edge to remove
			E_test.append(edge)
			indegrees[edge[0]] -= 1
			indegrees[edge[1]] -= 1
			full_edge_list[i] = None
			L_test -= 1
			if L_test == 0:
				# Success condition, we are done removing edges
				print("Success")
				break
		else:
			# Invalid edge, reset the adj matrix removal
			adj_matrix[edge[0], edge[1]] = 1
			adj_matrix[edge[1], edge[0]] = 1
		temp_count += 1
	E_train = [edge for edge in full_edge_list if edge is not None]

	#the remaining edges make up the training set
	return (E_train, E_test, E_fake)

def calcIndegrees(edge_list, num_nodes):
	'''
	Calculates the indegrees of each node.
	Takes in the edge list, number of nodes
	Returns an array, where each index corresponds with that node. Contains
	the number of edges that link to that node
	'''
	indegrees = [0] * num_nodes
	for edge in edge_list:
		indegrees[edge[0]] += 1
		indegrees[edge[1]] += 1
	return indegrees

def toAdjMatrix(edge_list, num_nodes):
	'''
	Takes in an edge list and turns it into an adjacency matrix
	'''
	graph = np.zeros((num_nodes, num_nodes))
	for edge in edge_list:
		graph[edge[0], edge[1]] = 1
		graph[edge[1], edge[0]] = 1
	return graph

def main():
	lists = separate_edge_lists(EDGE_LIST_PATH)
	E_train = lists[0]
	E_test = lists[1]
	E_fake = lists[2]
	with open(WRITE_TRAINING_PATH, 'w') as train_file:
		for edge in E_train:
			train_file.write(str(edge[0]) + " " + str(edge[1]) + "\n")

	with open(WRITE_REMOVED_PATH, 'w') as removed_file:
		for edge in E_test:
			removed_file.write(str(edge[0]) + " " + str(edge[1]) + "\n")

	with open(WRITE_MISSING_PATH, 'w') as missing_file:
		for edge in E_fake:
			missing_file.write(str(edge[0]) + " " + str(edge[1]) + "\n")

if __name__ == '__main__':
	main()
