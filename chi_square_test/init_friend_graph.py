#!/usr/bin/env python

'''
Script to randomize the user social graph based on the
Greedy algorithm presented by La Fond and Neville.  The 
Number of friends each user has will remain constant after
the randomization procedure is complete.
'''

import random

#A UserNode class
class UserNode:

	def __init__(self, userID):
		self.id = userID
		self.numFriends = 0
		self.friends = set()

	def add_friend(self, friendID):
		self.friends.add(friendID)
		#this assumes the file will not contain any lines that include a user's own ID in their list of friends
		self.numFriends+=1

	def print_info(self):
		print("ID: " + self.id)
		print("User has " + str(self.numFriends) + " friends")
		for i in range(0,self.numFriends):
			print(self.friends.pop())

	def __eq__(self, other):
		return self.id == other.id

	def __hash__(self):
		return hash(self.id)

#read in friends from a file where each line is a userID followed by the 
#userIDs of their friends, separated by tabs
def init_friend_graph(pathname):
	file = open(pathname, "r")

	users = {}

	for line in file:
		elements = line.strip().split("\t")
		friends = []
		user = elements[0]
		for next_friend in elements[1:]:
			friends.append(next_friend)

		users[user] = friends

	return users