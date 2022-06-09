import csv
import numpy as np
import sys
import re
import math
from pandas import read_csv

BANNED_DATA = ["Unrated", math.nan]

def load_data(filename):
	# load data, dropping header and weird trailing comma in CSV
	data = read_csv('ramen-ratings.csv').to_numpy()[:, 1:-1]
	row_list = []
	for i in range(data.shape[0]):
		row = data[i, :]
		if len(set(BANNED_DATA).intersection(set(row))) > 0:
			continue
		row_list.append(np.array(process_row(row)))
	return np.array(row_list, dtype=object)

def process_row(row):
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
	"""
	Converts first four features to continuous.
	"""
	for i in range(2):
		X[:, i] = convert_list_column_to_continuous(X, i)

	X[:, 2] = convert_list_column_to_continuous(X, 2)
	X[:, 3] = convert_list_column_to_continuous(X, 3)
	
	return X

def convert_column_to_continuous(data, col_num):
	"""
	Returns provided data with specified column converted to continuous frequency variable.
	This is basically a measure of how popular the words in the title/brandname/etc are in the overall dataset.
	"""
	new_col = data[:,col_num]

	word_list = []
	for i in range(len(new_col)):
		word_list.append(new_col[i])

	word_list = np.array(word_list)

	unique, counts = np.unique(word_list, return_counts=True)

	mean, std = calc_mean_std(counts)
	counts = z_score(counts, mean, std)

	freq_dict = {}

	for i in range(len(unique)):
		freq_dict[unique[i]] = counts[i]
	
	for i in range(len(new_col)):
		freq_sum = 0
		freq_sum += freq_dict[new_col[i]]
		new_col[i] = freq_sum

	return new_col / np.max(new_col)

def convert_list_column_to_continuous(data, col_num):
	"""
	Returns provided data with specified column converted to continuous frequency variable.
	This is basically a measure of how popular the words in the title/brandname/etc are in the overall dataset.
	"""
	new_col = data[:,col_num]

	word_list = []
	for i in range(len(new_col)):
		for word in new_col[i]:
			word_list.append(word)

	word_list = np.array(word_list)

	unique, counts = np.unique(word_list, return_counts=True)

	mean, std = calc_mean_std(counts)
	counts = z_score(counts, mean, std)

	freq_dict = {}

	for i in range(len(unique)):
		freq_dict[unique[i]] = counts[i]
	
	for i in range(len(new_col)):
		freq_sum = 0
		for word in new_col[i]:
			try:
				freq_sum += freq_dict[word]
			except:
				pass
		new_col[i] = freq_sum

	return new_col / np.max(new_col)

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
    std = np.std(X.astype(float), axis=0, ddof=1)
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
	

