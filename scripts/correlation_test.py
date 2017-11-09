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
import copy

RAND_NUM = 10

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
        edge_set = get_edge_set(user_ratings, graph)
        rand_graphs = get_random_edge_graphs(edge_set, RAND_NUM)
        user_ratings_dict = {}
        for user in user_ratings:
            user_ratings_dict[user[0]] = user[1]
        return [calc_chi_sq(user_ratings_dict, random_graph_x) for random_graph_x in rand_graphs]
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

def sort(edge):
    edge_list = sorted([e for e in edge])
    result = (edge_list[0], edge_list[1])
    return result

def get_edge_set(user_ratings, graph):
    '''
    Create the set of birectional edges in the graph of users who have reviewed a particular restaurant.
    user_ratings is a list of tuples (user, rating)
    graph is a list of lists, the first entry in each list is a user, each subsequent entry is a friend of the user

    return the set of bidirectional edges to be randomized
    '''

    #get the set of users in the user_ratings list
    user_set = set()
    for u in user_ratings:
        user_set.add(u[0])

    #create the original set of edges
    edges = set()
    #for each user that has reviewed this restaurant
    for user in user_ratings:
        start = user[0]    
        #iterate over this user's friends     
        for friend in graph[start]:
            #check to see if this friend has reviewed the restaurant
            if friend in user_set:
                e = sort((start, friend))
                edges.add(e)

    return edges

def is_valid_swap(edge_one, edge_two, edges):
    '''
    determine if edge two is a valid swap for edge one
    edge_one and edge_two are tuples representing bidirectional edges
    edges is the set of edges
    '''
    new_edge_one = sort((edge_two[0],edge_one[1]))
    new_edge_two = sort((edge_one[0],edge_two[1]))

    if new_edge_one in edges:
        return False
    if new_edge_two in edges:
        return False

    return True



def get_random_edge_graphs(edges, num_permutations):
    '''
    Takes in the set of bidrectional edges representing the edges in the graph of users who have reviewed this restaurant
    Return a list of user->list of friends dictionaries, each representing a randomized adjacency list
    '''

    #the list of randomized adjacency lists to return
    randomized_graphs = []

    for i in range(num_permutations):
        randomized_edges = list(edges)
        #create a copy of the edges set
        e = copy.deepcopy(edges)
        #iterate through each of the edges in the list
        for edge in randomized_edges:
            #if this edge has already been swapped out of the randomized set, skip it
            if edge not in e:
                continue
            found_edge_to_swap = False
            counter = 0
            while found_edge_to_swap is False and counter < len(randomized_edges):
                #somehow still getting numbers up to and including the upper bound, specification says this shouldn't be so
                rand_index = random.randint(0, len(randomized_edges)-1)
                #if this edge is still in the set and is an appropriate edge to swap, swap with it
                rand_edge = randomized_edges[rand_index]
                if rand_edge in e and is_valid_swap(rand_edge, edge, e):
                    edge_one = (rand_edge[0], edge[1])
                    edge_two = (edge[0], rand_edge[1])
                    e.remove(edge)
                    e.remove(rand_edge)
                    e.add(edge_one)
                    e.add(edge_two)
                    found_edge_to_swap = True
                else:
                    counter+=1

        adjlist = {}

        #use these randomly generated edges to create an adjacency list
        for edge in e:
            userid = edge[0]
            if userid in adjlist:
                #add end of this edge to the user in the adjacency list
                friends = adjlist[userid]
                friends.append(edge[1])
                adjlist[userid] = friends
            else:
                #create a new entry in the adjacency list for this new user
                adjlist[edge[0]] = [edge[1]]

            #do this again for the opposite direction, since the edges are bidirectional
            userid = edge[1]
            if userid in adjlist:
                #add end of this edge to the user in the adjacency list
                friends = adjlist[userid]
                friends.append(edge[0])
                adjlist[userid] = friends
            else:
                #create a new entry in the adjacency list for this new user
                adjlist[edge[1]] = [edge[0]]

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
        if user1 not in graph or user2 not in graph[user1]:
            if user_ratings_map[user1] == user_ratings_map[user2]:
                c += 1
            else:
                d += 1
        else:
            if user_ratings_map[user1] == user_ratings_map[user2]:
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
