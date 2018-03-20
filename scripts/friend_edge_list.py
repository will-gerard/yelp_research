'''
Writes an edge text file for use with the GEM implementation of SDNE
'''

from init_friend_graph import init_friend_graph


def main():
    friend_graph = init_friend_graph('../data/yelpFriends.txt')
    index_to_user = list(friend_graph.keys())
    user_to_index = {k: v for v, k in enumerate(index_to_user)}
    V = len(index_to_user)
    edges = set()
    for user1, friend_list in friend_graph.items():
        for user2 in friend_list:
            idx1 = user_to_index.get(user1)
            idx2 = user_to_index.get(user2)
            if (idx1 is not None and idx2 is not None):
                edges.add(tuple(sorted([idx1, idx2])))

    edges = sorted(edges)
    with open('../data/friend_edge_list.txt', 'w') as f:
        for edge in edges:
            f.write(str(edge[0]) + " " + str(edge[1]) + "\n")

if __name__ == "__main__":
    main()
