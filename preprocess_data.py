import csv
from heapq import heapify
import numpy as np
import sys
import re
import math

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

def convert_X_to_continuous(X):
	continuous_X = None
	N, D = X.shape
	for d in range(D):
		feature_column = X[:,d]
		#called for every column in X
		cont_f_column = convert_column_to_continuous(feature_column)
		#cont_f_column = feature_column #for testing
		cont_f_column = np.reshape(cont_f_column, (N, 1))
		print(cont_f_column.shape)
		if continuous_X is None:
			continuous_X = cont_f_column
		else:
			continuous_X = np.concatenate((continuous_X, cont_f_column), axis=1)
	return continuous_X

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

def split_X_Y(data):
    X = data[:,:-1]
    Y = data[:,-1]
    return X, Y

def split_train_valid(data, train_percent=0.66):
	data_dim = data.shape
	total_rows = data_dim[0]
	index = math.ceil(train_percent * total_rows)
	
	train = data[:index]
	valid = data[index:]
	
	return train, valid

def calc_mean_std(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    return mean, std

def z_score(X, mean, std):
    N = len(X)
    norm_X = np.divide(np.subtract(X, mean), std)
    return norm_X

def add_dummy(X):
	N = len(X)
	dummy = np.ones(N)
	return np.insert(X, 0, dummy, axis=1)

if __name__ == '__main__':
	filename = 'ramen-ratings.csv'
	data = load_data(filename)
	#print(data[0])
	X, Y = split_X_Y(data)
	#print(X[0], "\n", Y[0])
	continuous_X = convert_X_to_continuous(X)

	discrete_X_train, discrete_X_valid = split_train_valid(X)
	cont_X_train, cont_X_valid = split_train_valid(continuous_X)
	Y_train, Y_valid = split_train_valid(Y)

	mean, std = cont_X_train, cont_X_valid
	

	#for testing
	'''X = np.array([[1, 2, 3],
					[4, 5, 6]])'''
	

