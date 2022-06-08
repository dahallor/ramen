import csv
import numpy as np
import sys
import re
import math
import pdb

def load_data(filename):
	data = []
	with open(filename, encoding="utf-8") as file:
		reader = csv.reader(file, delimiter=',')
		next(reader)#skip header
		for row in reader:
			#print(row)
			if row[5] == "Unrated": #discard 3 Unrated rows
				continue
			row = process_row(row)
			data.append(row)
	data = np.array(data, dtype=object)
	return data

def process_row(row):
	row = row[1:6]
	row[4] = float(row[4])
	for i in range(0,4):
		if i < 2:
			row[i] = process_phrase(row[i])
		else:
			row[i] = [row[i]]
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
		#print(cont_f_column.shape)
		if continuous_X is None:
			continuous_X = cont_f_column
		else:
			continuous_X = np.concatenate((continuous_X, cont_f_column), axis=1)
	return continuous_X

def convert_column_to_continuous(column):
	words = {}
	for lst_of_words in column:
		for word in lst_of_words:
			#print(word)
			if word in words:
				words[word] = words[word] + 1
			else:
				words[word] = 1
	total_words = sum(words.values())
	#print(total_words)
	word_frequencies = {key: value / total_words for key, value in words.items()}

	continuous = []
	for lst_of_words in column:
		sum_frequencies = 0
		for word in lst_of_words:
			sum_frequencies += word_frequencies[word]
		#print(sum_frequencies)
		continuous.append(sum_frequencies)
	return np.array(continuous)

def shuffle_data(data, seed=0):
	np.random.seed(seed)
	np.random.shuffle(data)

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

def rounded_Y(Y):
	new_Y = []
	for y in Y:
		new_Y.append(round(y))
	return np.array(new_Y)

if __name__ == '__main__':
	#You can use this for testing the preprocess_data functions
	'''X = np.array([[1, 2, 3],
					[4, 5, 6]])'''
	filename = "ramen-ratings.csv"
	data = load_data(filename)
	
	X, Y = split_X_Y(data)
	print(X[0])
	continuous_X = convert_X_to_continuous(X)
	print(continuous_X[0])
	

