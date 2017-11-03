#!/usr/bin/env python

'''
Determines the significance of correlation between friendship in a Yelp friend
graph and equal ratings. Reads the Yelp friends graph and the user ratings map and
performs randomization on the data. A chi square value is calculated on the data
set as a measure of correlation. This value is calculated for each individual restaurant
when possible and summed together to form a larger chi square statistic.

Restaurants where expected value in the chisq contingency table are ignored.

This is repeated RAND_NUM times on randomized data to create a distribution,
and then the percentile of the actual chi square statistic for the real data set
is determined based on the randomized data to determine significance.

Argument 1: 'score' or 'edge' to determine if scores should be randomized or
    edges in the friend graph
'''

from init_friend_graph import init_friend_graph
from collections import OrderedDict
import json
import random
from random import shuffle
from itertools import combinations
import sys

RAND_NUM = 100

def main():
    rest_user_ratings_map = read_restaurant_by_user_ratings()
    friend_graph = init_friend_graph('../data/yelpFriends.txt')

    # A list of lists, where each list corresponds to a restaurant subgraph
    # Each list contains RAND_NUM chi sq values generated from permuting the graph
    random_chisq_values = []

    # Running sum of the chi sq values for each restaurant of the real graph
    real_chisq_sum = 0

    counter = 0

    for r_id in rest_user_ratings_map:
        user_ratings = rest_user_ratings_map[r_id]
        user_ratings = remove_dups(user_ratings)

        # generate random chisq values first:
        chisq_vals = gen_random_chisq_vals(user_ratings, friend_graph)
        print("calculated " + str(counter) + " of " + str(len(rest_user_ratings_map)))
        counter+=1
        random_chisq_values.append(chisq_vals)

        # calculate chisq value for actual graph
        real_chisq_sum += calc_chi_sq(dict(user_ratings), friend_graph)

    # list of chisq sums for RAND_NUM trials
    sum_rand_chisq = [sum(i) for i in zip(*random_chisq_values)]
    sum_rand_chisq.sort()

    print(sum_rand_chisq)
    print(real_chisq_sum)
    # calculate what percentile the real sum is in compared to the distribution of random
    percentile = 1
    for i in range(len(sum_rand_chisq)):
        if sum_rand_chisq[i] > real_chisq_sum:
            percentile = i/len(sum_rand_chisq)
            break
    print("Percentile: " + str(percentile))



def read_restaurant_by_user_ratings():
    '''
    Reads user ratings grouped by restaurants
    '''
    with open('../data/restaurants') as restaurant_f:
        user_ratings_map = json.load(restaurant_f)
        return user_ratings_map


def remove_dups(user_ratings):
    '''
    Resolves multiple reviews from the same user. Only keeps the last review score
    found from the user. Takes in a list of tuples (user_id, rating) and returns a
    modified list of tuples.
    '''
    d = OrderedDict(user_ratings)
    return list(d.items())

def gen_random_chisq_vals(user_ratings, graph):
    '''
    Generates random chisq values for a particular restaurant, based on whether
    command line argument to randomize scores or graph edges. Defaults to score if
    no argument is supplied
    '''

    if len(sys.argv) > 1:
        rand_type = sys.argv[1]
    else:
        rand_type = 'score'

    if rand_type == 'score':
        rand_user_ratings = get_random_score_ratings(user_ratings, RAND_NUM)
        return [calc_chi_sq(x, graph) for x in rand_user_ratings]
    elif rand_type == 'edge':
        rand_edges = get_random_edges(user_ratings, graph, RAND_NUM)
        user_ratings_dict = {}
        for user in user_ratings:
            user_ratings_dict[user[0]] = user[1]
        return [calc_chi_sq(user_ratings_dict, random_graph_x) for random_graph_x in rand_edges]
        #return calc_chi_sq(user_ratings, rand_edges[1])
    else:
        raise Exception("Argument 1 must be either 'score' or 'edge'")


def get_random_score_ratings(user_ratings_list, num):
    '''
    Takes in a list of (users, ratings) and returns a list of num dicts with the
    ratings permuted in each one
    '''
    length = len(user_ratings_list)
    ans = []
    for i in range(num):
        user_list = [entry[0] for entry in user_ratings_list]
        rating_list = [entry[1] for entry in user_ratings_list]
        shuffle(rating_list)
        permute_dict = dict(zip(user_list, rating_list))
        ans.append(permute_dict)
    return ans

def reverse(edge):
    start = edge[0]
    end = edge[1]
    result = (end, start)
    return result

def get_random_edges(user_ratings, graph, num_permutations):
    '''
    Takes in user ratings for a particular restaurant, the friend graph, and the number
    randomly generated permutations to create
    Return a list of user->list of friends dictionaries
    '''

    #create the original set of edges
    edges = []
    for user in user_ratings:
        start = user[0]         
        for friend in graph[start]:
            e = (start, friend)
            edges.append(e)

    #the list of randomized adjacency lists to return
    randomized_graphs = []

    for i in range(num_permutations):
        randomized_edges = list(edges)
        #pick two random edges from the list and swap their endpoints
        for i in range(len(edges)):
            first = random.choice(randomized_edges)
            second = random.choice(randomized_edges)
            while second is first:
                second = random.choice(randomized_edges)
            new_first = (first[0], second[1])
            new_second = (second[0], first[1])
            randomized_edges.remove(first)
            randomized_edges.remove(second)
            #randomized_edges.remove(reverse(first))
            #randomized_edges.remove(reverse(second))
            randomized_edges.append(new_first)
            randomized_edges.append(new_second)
            #randomized_edges.append(reverse(new_first))
            #randomized_edges.append(reverse(new_second))

        adjlist = {}

        #use these randomly generated edges to create an adjacency list
        for e in randomized_edges:
            userid = e[0]
            if userid in adjlist:
                #add end of this edge to the user in the adjacency list
                friends = adjlist[userid]
                friends.append(e[1])
                adjlist[userid] = friends
            else:
                #create a new entry in the adjacency list for this new user
                adjlist[e[0]] = [e[1]]
        randomized_graphs.append(adjlist)

    return randomized_graphs


def calc_chi_sq(user_ratings_map, graph):
    '''
    Takes a mapping from user_id => ratings and adjacency list graph. Calculates
    the chi square value. If an exp value is 0, returns 0 since the exp value
    will be 0 for all permutations of graphs/scores
    '''
    
    users = list(user_ratings_map.keys())
    user_pairs = list(combinations(users, 2))
    a = 0 # friends and share rating
    b = 0 # friends and don't share rating
    c = 0 # not friends and share rating
    d = 0 # not friends and don't share rating
    for user1, user2 in user_pairs:
        if user2 in graph[user1]:
            if user_ratings_map[user1] == user_ratings_map[user2]:
                a += 1
            else:
                b += 1
        else:
            if user_ratings_map[user1] == user_ratings_map[user2]:
                c += 1
            else:
                d += 1
    num = ((a*d-c*b)**2) * (a+b+c+d)
    denom = (a+b) * (c+d) * (b+d) * (a+c)
    if denom == 0:
        return 0
    return num/denom



if __name__ == "__main__":
    main()
