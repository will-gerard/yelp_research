#!/usr/bin/env python

from init_friend_graph import init_friend_graph
import json
from random import shuffle
from itertools import combinations


def main():
    restaurant_user_ratings_map = read_restaurant_by_user_ratings()
    friend_graph = init_friend_graph('../data/yelpFriends.txt')
    


def read_restaurant_by_user_ratings():
    '''
    Reads user ratings grouped by restaurants
    '''
    with open('../data/restaurants') as restaurant_f:
        user_ratings_map = json.load(restaurant_f)
        return user_ratings_map


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


def calc_chi_sq(user_ratings_map, graph):
    '''
    Takes a mapping from user_id => ratings and adjacency list graph. Calculates
    the chi square value
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
    return num/denom



if __name__ == "__main__":
    main()
