'''
Checks if there are any disconnected nodes in the edge list found at PATH
AKA if there are node indices that don't exist in the edge list b/c nothing connects to it
'''

PATH = "../data/edgelists/user_training.edgelist"

with open(PATH) as f:
    node_num, edge_num = [int(x) for x in f.readline().strip().split()]
    all_nums = set()
    for line in f:
        node1, node2 = [int(x) for x in line.strip().split()]
        all_nums.add(node1)
        all_nums.add(node2)
    for i in range(node_num):
        if i not in all_nums:
            print(i)
    all_nums = list(all_nums)
    all_nums = sorted(all_nums)
    # #print(all_nums)
