#!/usr/bin/env python

'''
Store the two similarity matrices necessary for the graph embedding:
1. Friend adjacency matrix V x V (boolean)
2. User-word usage V x W (sparse int)
'''
from collections import OrderedDict
from init_friend_graph import init_friend_graph
from itertools import islice
import json
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

restaurants = {}

def main():
	print("Initializing Friend Graph...")
	friend_graph = init_friend_graph('../data/yelpFriends.txt')
	index_to_user = list(friend_graph.keys())
	user_to_index = {k: v for v, k in enumerate(index_to_user)}
	V = len(index_to_user)
	friend_matrix = np.full((V,V), False)
	for user1, friend_list in friend_graph.items():
		for user2 in friend_list:
			idx1 = user_to_index.get(user1)
			idx2 = user_to_index.get(user2)
			if (idx1 is not None and idx2 is not None):
				friend_matrix[idx1, idx2] = True
				friend_matrix[idx2, idx2] = True


	user_data_directory = os.getcwd() + "/../data/yelp_users/"
	print("Reading reviews from each user...")
	for user in os.listdir(user_data_directory):
		filename = os.getcwd() + "/../data/yelp_users/" + user
		readUserReviews(filename)

	word_labels, user_word_matrix = generate_user_word_matrix(index_to_user)
	print(user_word_matrix.shape)
	result = {
		'user_labels': index_to_user,
		'word_labels': word_labels,
		'friend_matrix': friend_matrix,
		'user_word_matrix': user_word_matrix,
	}
	print("Generating pickle object...")
	pickle_object = open("../data/autoencoder_sim_matrices.pkl", "wb")
	pickle.dump(result, pickle_object)
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


def generate_user_word_matrix(index_to_user):
	'''
	Takes a list mapping index in table to user_id
	Returns tuple of (word feature labels, user-word binary sim matrix)
	'''
	tfidf_vectorizer = TfidfVectorizer()
	user_reviews = {} # dict of user => concatenation of all his reviews

	# concatenate each user's reviews together into user_reviews
	for rest_id, r in restaurants.items():
		#first filter the list of reviews to remove duplicates from the same user
		reviews = remove_dups(r)

		for i, review_tuple in enumerate(reviews):
			# insert/concatenate user review into dict
			if review_tuple[0] in user_reviews:
				user_reviews[review_tuple[0]] += " " + review_tuple[1]
			else:
				user_reviews[review_tuple[0]] = review_tuple[1]

	# add each user's reviews into the corpus so that the order of the reviews
	# is the same as the order of users
	corpus = [user_reviews[user] for user in index_to_user]


	tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
	tfidf_matrix[tfidf_matrix > 0] = 1 ###### If we only care about presence/absence of words

	return (tfidf_vectorizer.get_feature_names(), tfidf_matrix)


def remove_dups(user_ratings):
	'''
	Resolves multiple reviews from the same user. Only keeps the last review score
	found from the user. Takes in a list of tuples (user_id, rating) and returns a
	modified list of tuples.
	'''
	d = OrderedDict(user_ratings)
	return list(d.items())


if __name__ == "__main__":
	main()
