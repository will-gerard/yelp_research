#!/usr/bin/env python

'''
Script to parse through all yelp users and create a mapping
from restaurant_id => [user ids, ratings] for
easier access. Writes to 'restaurants' file as json
'''


import json
import os
from itertools import islice

'''
dict to hold restaurants mapping to user reviews:
{
    restaurnt_id => [
        (user_id, rating), ...
    ]
    ...
}
'''
restaurants = {}

def main():
    yelp_user_data_dir = os.getcwd() + "/../data/yelp_users/"
    for user in os.listdir(yelp_user_data_dir):
        filename = os.getcwd() + "/../data/yelp_users/" + user
        readUserFile(filename)
    writeRestaurants()



def readUserFile(filename):
    '''
    Takes in f, a file handle
    Reads 5 lines at a time. Line 0 is the restaurant id,
    3 is the restaurant's rating.
    Updates restaurant dict.
    '''
    global restaurants
    with open(filename, 'r') as f:
        user_id = f.readline().strip()
        while True:
            lines = list(islice(f, 5))
            if not lines:
                # EOF reached
                break
            restaurant_id = lines[0].strip()
            rating = lines[3].strip()
            if restaurant_id in restaurants:
                restaurant_list = restaurants[restaurant_id]
                restaurant_list.append((user_id, rating))
            else:
                restaurants[restaurant_id] = [(user_id, rating)]


def writeRestaurants():
    restaurant_file = os.getcwd() + "/../data/restaurants"
    with open(restaurant_file, 'w') as f:
        json.dump(restaurants, f)
        ## writing to file line by line if json is not preferred?
        # for k, v in restaurants.items():
        #     f.write(k + "\n")
        #     for user in v:
        #         f.write(user[0] + "\n")
        #         f.write(user[1] + "\n")
        #     f.write("\n")


if __name__ == "__main__":
    main()
