#!/usr/bin/env python

'''
Determines the significance of correlation between friendship in a Yelp friend
graph and the cosine similarity between the tfidf vectors of the words in the reviews for a particular restaurant. 
Reads the Yelp friends graph and the pregenerated cosine similaity matrix and performs a randomization test on the data. 
A chi square value is calculated on the data set which measures the strength of this correlation. 
This value is calculated for each individual restaurant and the results are summed 
together to form a larger chi square statistic.

Restaurants where expected value in the chisq contingency table is 0 are ignored.

This is repeated RAND_NUM times on randomized data to create a distribution,
and then the percentile of the actual chi square statistic for the real data set
is determined based on the randomized data to determine significance.

Argument 1: RAND_NUM

Argument 2: 'score' or 'edge' to determine if scores or edges should be randomized 
in the friend graph.
'''

from correlation_test import (remove_dups_to_dict, get_user_combos_from_ratings_map,
    get_random_score_ratings, get_random_edge_graphs, get_edge_set)
from init_friend_graph import init_friend_graph
from itertools import combinations
import pickle
import random
from random import shuffle
import sys

RAND_NUM = int(sys.argv[1])


def main():
    rest_user_reviews_map = read_restaurant_by_user_reviews()
    num_restaurants = len(rest_user_reviews_map)
    friend_graph = init_friend_graph('../data/yelpFriends.txt')

    # A list of lists, where each list corresponds to a restaurant subgraph
    # Each list contains RAND_NUM chi sq values generated from permuting the graph
    random_chisq_values = []

    # Running sum of the chi sq values for each restaurant of the real graph
    real_chisq_sum = 0

    counter = 0

    for r_id in rest_user_reviews_map:
        user_reviews = rest_user_reviews_map[r_id]
        combinations = get_user_combos_from_ratings_map(user_reviews['users'])
        # generate random chisq values first:
        chisq_vals = gen_random_chisq_vals(user_reviews, friend_graph, combinations)
        #print("calculated " + str(counter) + " of " + str(num_restaurants))
        counter += 1
        random_chisq_values.append(chisq_vals)

        # calculate chisq value for actual graph
        real_chisq_sum += calc_chi_sq(user_reviews['users'], user_reviews['cos_matrix'], friend_graph, combinations)

    # list of chisq sums for RAND_NUM trials
    sum_rand_chisq = [sum(i) for i in zip(*random_chisq_values)]
    sum_rand_chisq.sort()

    for num in sum_rand_chisq:
        print(str(num))
    print(real_chisq_sum)
    # calculate what percentile the real sum is in compared to the distribution of random
    percentile = 1
    for i in range(len(sum_rand_chisq)):
        if sum_rand_chisq[i] > real_chisq_sum:
            percentile = i/len(sum_rand_chisq)
            break
    print("Percentile: " + str(percentile))


def read_restaurant_by_user_reviews():
    '''
    Reads user ratings grouped by restaurants
    '''
    with open('../data/cosSimilarityNoNullRestaurants.pkl', 'rb') as restaurant_f:
        user_reviews_map = pickle.load(restaurant_f)
        return user_reviews_map


def gen_random_chisq_vals(user_reviews, graph, combinations):
    '''
    user_reviews = {
        'users': {user_id: index, ....}
        'cos_matrix': 'cosine similarity matrix'
    }
    Takes in a dict of user_id to list of user_id friends,
    and a list of tuples of all combinations of users.
    Generates random chisq values for a particular restaurant, based on whether
    command line argument to randomize scores or graph edges. Defaults to score if
    no argument is supplied
    '''
    reviews_map = user_reviews['users']
    cos_matrix = user_reviews['cos_matrix']
    if len(sys.argv) > 2:
        rand_type = sys.argv[2]
    else:
        rand_type = 'score'

    if rand_type == 'score':
        rand_user_ratings = get_random_score_ratings(reviews_map, RAND_NUM)
        return [calc_chi_sq(x, cos_matrix, graph, combinations) for x in rand_user_ratings]
    elif rand_type == 'edge':
        edge_set = get_edge_set(reviews_map, graph)
        rand_graphs = get_random_edge_graphs(edge_set, RAND_NUM)
        return [calc_chi_sq(reviews_map, cos_matrix, random_graph_x, combinations) for random_graph_x in rand_graphs]
    else:
        raise Exception("Argument 1 must be either 'score' or 'edge'")


def calc_chi_sq(reviews_map, cos_matrix, graph, user_pairs):
    '''
    Takes a dict of user_id => reviews_index, cosine similarity matrix,
    an adjacency list graph, and a list of
    tuples of all user pairs for the restaurant.
    Calculates the chi square value. If an exp value
    is 0, returns 0 since the exp value will be 0 for all permutations of graphs/scores
    '''
    a = 0 # friends and share rating
    b = 0 # friends and don't share rating
    c = 0 # not friends and share rating
    d = 0 # not friends and don't share rating
    for user1, user2 in user_pairs:
        user1_index = reviews_map[user1]
        user2_index = reviews_map[user2]
        if user1 not in graph or user2 not in graph[user1]:
            if cos_matrix[user1_index][user2_index]:
                c += 1
            else:
                d += 1
        else:
            if cos_matrix[user1_index][user2_index]:
                a += 1
            else:
                b += 1
    num = ((a*d-c*b)**2) * (a+b+c+d)
    denom = (a+b) * (c+d) * (b+d) * (a+c)
    if denom == 0:
        return 0
    return num/denom


if __name__ == "__main__":
    main()
