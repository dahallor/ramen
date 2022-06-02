import csv
from heapq import heapify
import numpy as np
import sys
import re

def load_data(filename):
	data = []
	with open(filename) as file:
		reader = csv.reader(file, delimiter=',')
		next(reader)#skip header
		for row in reader:
			if row[5] == "Unrated": #discard 3 Unrated rows
				continue
			row = process_row(row)
			data.append(row)
	data = np.array(data, dtype=object)
	return data

def process_row(row):
	row = row[1:6]
	row[4] = float(row[4])
	for i in range(0,2):
		row[i] = process_phrase(row[i])
	return row

def process_phrase(phrase):
	words = phrase.lower().split(" ")
	new_words = []
	for word in words:
		new_word = re.sub("[^A-Za-z0-9]+", "", word)
		if new_word != "":
			new_words.append(new_word)
	return new_words

#discretize the data into quartiles
#I don't know how to do this...
def convert_column_to_continuous(column):
	words = {}
	for lst_of_words in column:
		for word in lst_of_words:
			if word in words:
				words[word] = words[word] + 1
			else:
				words[word] = 1
	#sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)
	#for i in sorted_words:
	#	print(i[0], i[1])
	total_words = sum(words.values())
	print(total_words)


if __name__ == '__main__':
	filename = 'ramen-ratings.csv'
	data = load_data(filename)
	convert_column_to_continuous(data[:,0])