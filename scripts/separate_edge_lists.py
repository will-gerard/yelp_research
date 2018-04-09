'''
Separates an edge list into training, missing, and removed lists for AUROC link prediction tests:
Takes in 5 command line parameters:

1: file path to original edge list
2: file path for where to write the training edge list
3: file path for where to write the removed edge list
4: file path for where to write the missing edge list

Ex:
python3 separate_edge_lists.py ../data/friend_edge_list.txt ../data/edgelists/user_training.edgelist ../data/edgelists/user_removed.edgelist ../data/edgelists/user_missing.edgelist
'''
import random
import math
import sys

EDGE_LIST_PATH = sys.argv[1]
WRITE_TRAINING_PATH = sys.argv[2]
WRITE_REMOVED_PATH = sys.argv[3]
WRITE_MISSING_PATH = sys.argv[4]

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
	full_edge_set = set()
	with open(full_edgelist_path) as fp:
		num_nodes, num_edges = [int(x) for x in next(fp).split()]

		for line in fp:
			first, second = [int(x) for x in line.split()]
			e = (first, second)
			full_edge_set.add(e)

	#get the number of elements in the testing set:
	L_train = math.ceil(training_percentage * num_edges)
	L_test = num_edges - L_train

	#generate L random nonexistent links
	E_fake = []
	count = 0
	while count < L_test:
		#pick two random nodes
		nodes = random.sample(range(1, num_nodes), 2)
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
	full_edge_list = list(full_edge_set)
	E_train = []
	E_test = []
	#randomly select the edges to remove
	#TODO: check to make sure removing these edges leaves the graph connected
	x = len(full_edge_list)
	indexes_to_remove = random.sample(range(1, x), L_test)
	for i in indexes_to_remove:
		edge = full_edge_list[i]
		E_test.append(edge)

		full_edge_list[i] = None

	for edge in full_edge_list:
		if edge is not None:
			E_train.append(edge)

	#the remaining edges make up the training set
	return (E_train, E_test, E_fake)

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
