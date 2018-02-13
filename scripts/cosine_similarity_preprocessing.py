#!/usr/bin/env python

'''
Script to parse through all yelp users and create a mapping
from restaurant_id => [cosine similarity matrix, userid -> index into matrix dictionary] 
Writes to 'restaurants_cosine_similarity.json' file as json
'''

import json
import os
from itertools import islice
from itertools import combinations
from collections import OrderedDict

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from init_friend_graph import init_friend_graph

import pickle

restaurants = {}
threshold = 0.75

def main():
	print("Initializing Friend Graph...")
	friend_graph = init_friend_graph('../data/yelpFriends.txt')

	user_data_directory = os.getcwd() + "/../data/yelp_users/"
	
	print("Reading reviews from each user...")
	for user in os.listdir(user_data_directory):
		filename = os.getcwd() + "/../data/yelp_users/" + user
		readUserReviews(filename)

	restaurant_dict = generate_cosine_similarity_matrix()

	print("initial length of restaurant dictionary: " + str(len(restaurant_dict)))
	#filter out restaurants that will lead to null chi square values
	print("Filtering out null chi square value restaurants...")
	filtered_restaurants = {k:v for (k,v) in restaurant_dict.items()
		if not is_chi_square_null(v, friend_graph)}

	print("After filtering out null chi square restaurants: " + str(len(filtered_restaurants)))

	print("Generating pickle object...")
	pickle_object = open("cosSimilarityNoNullRestaurants.pkl", "wb")
	pickle.dump(filtered_restaurants, pickle_object)
	pickle_object.close()

	print("Done. Quitting...")


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


def generate_cosine_similarity_matrix():
	'''
	Take in the rest_id -> [(user_id_1, review_1), ...] dictionary.
	For each restaurant, generate a tfidf matrix of the corpus (all the reviews for that particular restaurant)
	Use the tfidf matrix to generate a cosine similarity matrix.
	'''
	global restaurants
	global threshold
	tfidf_vectorizer = TfidfVectorizer()
	#dictionary {resaurant_id - > {cos_matrix -> cos_matrix, users - > {user_id -> index_into_cos_matrix}}}
	restaurant_dictionary = {}

	for rest_id, r in restaurants.items():
		print("Generating dictionary for restaurant " + str(rest_id) + "...")
		#create the entry in restaurant_dictionary that will be added for this restaurant
		dict_entry = {}
		#create the corpus, a list of all the reviews for this restaurant
		corpus = []
		#dictionary of user -> index of corresponding row in matrix
		users = {}

		#first filter the list of reviews to remove duplicates from the same user
		reviews = remove_dups(r)

		for i, review_tuple in enumerate(reviews):
			#add the text of the review as a new document in the corpus
			corpus.append(review_tuple[1])
			#and add (this user_id -> current row in matrix) to the user dictionary 
			users[review_tuple[0]] = i

		#add the user indices to the restaurant dictionary value
		dict_entry['users'] = users

		tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
		#get the number of rows in the matrix
		num_rows = tfidf_matrix.shape[0]
		#this will be a 2D array representing the matrix of cosine similarity values
		cos_similarity_matrix = np.empty((num_rows, num_rows))
		#for each row, get the corresponding array of cosine similarity values
		# for i in range(0, num_rows):
		# 	cos_values_array = cosine_similarity(tfidf_matrix.getrow(i), tfidf_matrix)
		# 	#we don't care about the actual values, we just need 1 (similar) or 0 (not similar)
		# 	for x in range(0, cos_values_array.size):
		# 		if cos_values_array[0,x] >= threshold:
		# 			cos_values_array[0,x] = 1
		# 		else:
		# 			cos_values_array[0,x] = 0

		# 	#add the row to the cosine similarity matrix
		# 	cos_similarity_matrix[i] = cos_values_array

		#add the cosine similarity matrix to the restaurant dictionary value
		dict_entry['cos_matrix'] = cos_similarity_matrix

		#add this restaurant id and value to the restaurant dictionary
		restaurant_dictionary[rest_id] = dict_entry

	print("Finished generating restuarant dictionary.")
	return restaurant_dictionary

def remove_dups(user_ratings):
	'''
	Resolves multiple reviews from the same user. Only keeps the last review score
	found from the user. Takes in a list of tuples (user_id, rating) and returns a
	modified list of tuples.
	'''
	d = OrderedDict(user_ratings)
	return list(d.items())

def is_chi_square_null(restaurant_dictionary, graph):
	'''
	Takes a mapping from user_id => ratings and adjacency list graph. Checks if
	there are exp values of 0.
	'''

	global threshold

	cos_similarity_matrix = restaurant_dictionary['cos_matrix']
	user_to_index = restaurant_dictionary['users']

	users = list(user_to_index.keys())
	user_pairs = list(combinations(users, 2))
	a = 0 # friends and review text is similar
	b = 0 # friends and review text is not similar
	c = 0 # not friends and review text is similar
	d = 0 # not friends and review text is not similar
	for user1, user2 in user_pairs:
		user1_index = user_to_index[user1]
		user2_index = user_to_index[user2]
		if user1 not in graph or user2 not in graph[user1]:
			if cos_similarity_matrix[user1_index][user2_index] >= threshold:
				c += 1
			else:
				d += 1
		else:
			if cos_similarity_matrix[user1_index][user2_index] >= threshold:
				a += 1
			else:
				b += 1
	denom = (a+b) * (c+d) * (b+d) * (a+c)
	if denom == 0:
		return True
	else:
		return False 
	

if __name__ == "__main__":
	main()