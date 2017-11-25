#!/usr/bin/env python

'''
Reads the Yelp friends graph and restaurant user reviews. Filters out restaurants
that have null chi square subgraphs from the restaurant list, and writes the
remaining restaurant user reviews to equalRatingNoNullRestaurants

The chi square test being used here compares friendship and equal ratings.
'''

from init_friend_graph import init_friend_graph
from collections import OrderedDict
import json
import os
from random import shuffle
from itertools import combinations
import sys


def main():
    rest_user_ratings_map = read_restaurant_by_user_ratings()
    friend_graph = init_friend_graph('../data/yelpFriends.txt')

    filtered_restaurants = {k:v for (k,v) in rest_user_ratings_map.items()
        if not isSubgraphNull(v, friend_graph)}
    print(len(filtered_restaurants))
    restaurant_file = os.getcwd() + "/../data/equalRatingNoNullRestaurants.txt"
    with open(restaurant_file, 'w') as f:
        json.dump(filtered_restaurants, f)


def read_restaurant_by_user_ratings():
    '''
    Reads user ratings grouped by restaurants
    '''
    with open('../data/restaurants') as restaurant_f:
        user_ratings_map = json.load(restaurant_f)
        return user_ratings_map


def isSubgraphNull(user_ratings, friend_graph):
    '''
    Checks if the user reviews for this restaurant result in a chi square calculation
    that is null
    '''
    user_ratings = remove_dups(user_ratings)
    return is_chi_sq_null(dict(user_ratings), friend_graph)


def remove_dups(user_ratings):
    '''
    Resolves multiple reviews from the same user. Only keeps the last review score
    found from the user. Takes in a list of tuples (user_id, rating) and returns a
    modified list of tuples.
    '''
    d = OrderedDict(user_ratings)
    return list(d.items())


def is_chi_sq_null(user_ratings_map, graph):
    '''
    Takes a mapping from user_id => ratings and adjacency list graph. Checks if
    there are exp values of 0.
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
        return True
    else:
        return False


if __name__ == "__main__":
    main()
