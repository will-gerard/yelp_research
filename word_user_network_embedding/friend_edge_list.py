'''
Writes an edge text file for use with the GEM implementation of SDNE or node2vec
Run with `python3 friend_edge_list.py`

Generates 2 edge lists:
1. friend_edge_list.txt - edges between users
2. friend_word_edge_list.txt - edges between users, and between users and friends

Format of edge lists:
The first line contains two numbers, the # of nodes and # of edges
All subsequent lines also have two numbers, indicating edges between those nodes
'''

from collections import OrderedDict
from init_friend_graph import init_friend_graph
from itertools import islice
import nltk
import numpy as np
import os
import re
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
import Stemmer
import string

restaurants = {}
stemmer = Stemmer.Stemmer('en')
TFIDF_THRESHOLD = 0.125

def readUserReviews(filename):
    '''
    Takes in f, a file handle
    Reads 5 lines at a time. Line 0 is the restaurant id,
    line 1 is the content of the review.
    Updates restaurant dict. to contain rest_id -> (user_id, review)
    '''
    global restaurants
    with open(filename, 'r') as f:
        #the first line in the file is the user id, don't include it in the sets of 5 lines
        user_id = f.readline().strip()
        while True:
            lines = list(islice(f, 5))
            if not lines:
                # EOF reached
                break
            restaurant_id = lines[0].strip()
            review = lines[1].strip()
            if restaurant_id in restaurants:
                restaurant_list = restaurants[restaurant_id]
                restaurant_list.append((user_id, review))
                restaurants[restaurant_id] = restaurant_list
            else:
                restaurants[restaurant_id] = [(user_id, review)]

def filter_disconnected_nodes(friend_graph):
    '''
    Takes in a adjacency list. 'Main users' are the keys of the friend graph.
    Filters out main users whose friends are all NOT main users
    @param friend_graph: {user_id: [user_id, ...]}

    @return filtered_graph: {user_id: [user_id, ...]}
    '''
    filtered_friends = {}
    for main_user, friend_list in friend_graph.items():
        num_main_friends = len([x for x in friend_list if x in friend_graph])
        if num_main_friends != 0:
            filtered_friends[main_user] = friend_list
    return filtered_friends

def get_largest_subgraph(friend_graph):
    '''
    Takes in an adjacency list and finds the largest connected subgraph
    @param friend_graph: {user_id: [user_id, ...]}

    @return filtered_graph: {user_id: [user_id, ...]}
    '''
    index_to_user = list(friend_graph.keys())
    user_to_index = {k: v for v, k in enumerate(index_to_user)}
    num_nodes = len(index_to_user)
    graph = np.zeros((num_nodes, num_nodes))
    for user1, friend_list in friend_graph.items():
        for user2 in friend_list:
            idx1 = user_to_index.get(user1)
            idx2 = user_to_index.get(user2)
            if (idx1 is not None and idx2 is not None):
                graph[idx1, idx2] = 1
                graph[idx2, idx1] = 1
    N_components, component_list = connected_components(graph, directed=False)
    components_size = [np.sum(component_list == i) for i in range(N_components)]
    largest_graph_index = max(enumerate(components_size),key=lambda x: x[1])[0]
    return {k: v for k, v in friend_graph.items() if component_list[user_to_index[k]] == largest_graph_index}

def friend_edge_list(friend_graph):
    '''
    Takes in the adjacency list and returns an edge list, along with a
    dict from user id to its corresponding index and a list of user_ids matching their
    index

    @param friend_graph : {user_id : [user_id, user_id, ...]}
        User friends adjacency list

    @return (edge list tuples, dictionary of user_id to index, list of user_ids)
    '''
    friend_graph = filter_disconnected_nodes(friend_graph)
    friend_graph = get_largest_subgraph(friend_graph)
    index_to_user = list(friend_graph.keys())
    user_to_index = {k: v for v, k in enumerate(index_to_user)}
    edges = set()
    for user1, friend_list in friend_graph.items():
        for user2 in friend_list:
            idx1 = user_to_index.get(user1)
            idx2 = user_to_index.get(user2)
            if (idx1 is not None and idx2 is not None):
                edges.add(tuple(sorted([idx1, idx2])))
    edges = sorted(edges)
    return (edges, user_to_index, index_to_user)


def tokenize(text):
    text.translate(str.maketrans('','',string.punctuation))
    tokens = nltk.tokenize.casual_tokenize(text, preserve_case=False, reduce_len=True)
    tokens = [re.sub('\d+', 'NUM', s) for s in tokens]
    stems = [stemmer.stemWord(item) for item in tokens]
    return stems


def generate_user_word_matrix(index_to_user):
    '''
    Takes a list mapping index in table to user_id
    Returns tuple of (word feature labels, user-word binary sim matrix)
    '''
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english', min_df=3, lowercase=False)
    user_reviews = {} # dict of user => concatenation of all his reviews

    # concatenate each user's reviews together into user_reviews
    for rest_id, r in restaurants.items():
        #first filter the list of reviews to remove duplicates from the same user
        reviews = remove_dups(r)

        for i, review_tuple in enumerate(reviews):
            # insert/concatenate user review into dict
            review_text = review_tuple[1]
            if review_tuple[0] in user_reviews:
                user_reviews[review_tuple[0]] += " " + review_text
            else:
                user_reviews[review_tuple[0]] = review_text

    # add each user's reviews into the corpus so that the order of the reviews
    # is the same as the order of users
    corpus = [user_reviews[user] for user in index_to_user]

    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    # hacky way to zero out values efficiently with sparse arrays
    tfidf_matrix[tfidf_matrix > TFIDF_THRESHOLD] = -1
    tfidf_matrix[tfidf_matrix > 0] = 0
    tfidf_matrix[tfidf_matrix == -1] = 1
    tfidf_matrix.eliminate_zeros() # necessary to actually eliminate the zero from the sparse pattern
    # remove word features that are no longer important (columns that are zeros)
    words_non_zero = set(np.nonzero(tfidf_matrix)[1])
    tfidf_matrix = tfidf_matrix[:, sorted(words_non_zero)]
    names = tfidf_vectorizer.get_feature_names()
    names = [n for idx, n in enumerate(names) if idx in words_non_zero]
    return (names, tfidf_matrix)


def remove_dups(user_ratings):
    '''
    Resolves multiple reviews from the same user. Only keeps the last review score
    found from the user. Takes in a list of tuples (user_id, rating) and returns a
    modified list of tuples.
    '''
    d = OrderedDict(user_ratings)
    return list(d.items())


def word_user_edge_list(word_labels, user_word_matrix):
    '''
    @param word_labels: list of words where the index corresponds to the word
    @param user_word_matrix: scipy sparse array. First axis is the users where the id's match
        the user indices created in friend_edge_list. Second axis is the words which
        match the word_labels. Contains 1 or 0 for if the user's reviews contain the word

    @return: (edges, word_to_index). Edges is a list of tuples of indices.
        word_to_index maps the word to its new index
    '''
    word_to_index = {}
    user_size = user_word_matrix.shape[0]
    edges = set()
    cx = user_word_matrix.tocoo() # faster to iterate over coo sparse matrix
    for user_id, word_id, _ in zip(cx.row, cx.col, cx.data):
        idx1 = user_id
        idx2 = word_id + user_size
        edges.add(tuple(sorted([idx1, idx2])))
        word_to_index[word_labels[word_id]] = idx2
    edges = sorted(edges)
    return (edges, word_to_index)


def main():
    print("Initializing friend graph...")
    friend_graph = init_friend_graph('../data/yelpFriends.txt')

    friend_edges, user_to_index, index_to_user = friend_edge_list(friend_graph)

    # iterate through users and read in reviews
    user_data_directory = os.getcwd() + "/../data/yelp_users/"
    print("Reading reviews from each user...")
    for user in os.listdir(user_data_directory):
        filename = os.getcwd() + "/../data/yelp_users/" + user
        readUserReviews(filename)

    print("Generating word matrix...")
    word_labels, user_word_matrix = generate_user_word_matrix(index_to_user)
    print(len(word_labels))
    print(word_labels)
    users_size = len(user_to_index)
    friend_edges_size = len(friend_edges)

    word_edges, word_to_index = word_user_edge_list(word_labels, user_word_matrix)

    print("Writing edge lists...")
    # write the friend list
    with open('../data/friend_edge_list.txt', 'w') as f:
        f.write(str(users_size) + " " + str(friend_edges_size) + "\n")
        for edge in friend_edges:
            f.write(str(edge[0]) + " " + str(edge[1]) + "\n")

    # combine the edges for friends and words together:
    all_edges = friend_edges + word_edges
    nodes_size = users_size + len(word_to_index)
    edges_size = len(all_edges)
    with open('../data/friend_word_edge_list.txt', 'w') as f:
        f.write(str(nodes_size) + " " + str(edges_size) + "\n")
        for edge in all_edges:
            f.write(str(edge[0]) + " " + str(edge[1]) + "\n")

if __name__ == "__main__":
    main()
