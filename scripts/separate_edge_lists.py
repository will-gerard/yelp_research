import random

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
		for line in fp:
			first, second = [int(x) for x in line.split()]
			e = (first, second)
			full_edge_set.add(e)

	#get the number of elements in the testing set:
	num_edges = len(full_edge_set)
	L_train = training_percentage * num_edges
	L_test = num_edges - L_train

	#TODO: make this number a parameter
	num_nodes = 10674

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
			count++

	#select L random edges to remove
	#get the full list of the edges, we will remove some to create the testing list
	E_train = list(full_edge_set)
	E_test = []
	#randomly select the edges to remove
	#TODO: check to make sure removing these edges leaves the graph connected
	indexes_to_remove = random.sample(range(1, len(full_edge_list)), L_test)
	for i in indexes_to_remove:
		edge = E_train.pop(i)
		E_test.append(edge)

	#the remaining edges make up the training set
	return (E_train, E_test, E_fake)