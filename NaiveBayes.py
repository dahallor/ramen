from itertools import count
import numpy as np
import random
import math
from preprocess_data import *
from valid_statistics import *

# training the model on training set
class NaiveBayes:
	def __init__(self, discrete_X_train, discrete_Y_train, discrete_X_valid, discrete_Y_valid):
		self.discrete_X_train = self.prepare_naive_bayes(discrete_X_train)
		self.discrete_Y_train = discrete_Y_train
		self.discrete_X_valid = self.prepare_naive_bayes(discrete_X_valid)
		self.discrete_Y_valid = discrete_Y_valid

	def prepare_naive_bayes(self, X):

		feature_list = []
		for row in range(X.shape[0]):
			sub_list = []
			for feature in range(len(X[row, :])):
				sub_list.extend(X[row, feature])
		
			feature_list.append(np.array(sub_list))

		return np.array(feature_list, dtype=object)


	def get_cond_prob(self, token_freq, label_token_freq, label_prior):
		probability = ((label_token_freq+1) / (token_freq+ 1)) * label_prior
		return probability

	def get_pred(self, observation):
		max_prob = -math.inf
		pred = -1
		for label in self.label_freqs.keys():
			probs = []
			for token in observation:
				token_freq = self.token_freqs[token] if (token in self.token_freqs.keys()) else 0
				label_token_freq = self.class_token_freqs[label][token] if (token in self.class_token_freqs[label].keys()) else 0
				prob = self.get_cond_prob(token_freq, label_token_freq, self.label_freqs[label]/self.total_count)

				# log prob
				probs.append(math.log(prob))

			total_prob = sum(probs)

			if total_prob > max_prob:
				max_prob = total_prob
				pred = label

		return pred
				

	def get_preds(self, X):
		preds = np.zeros(X.shape[0])
		for i in range(X.shape[0]):
			preds[i] = self.get_pred(X[i])

		return preds

	def train(self):

		self.class_token_freqs = {}
		self.token_freqs = {}
		self.label_freqs = {}
		self.total_count = 0

		for i in range(len(self.discrete_Y_train)):
			label = self.discrete_Y_train[i]
			tokens = self.discrete_X_train[i]
			
			if label not in self.class_token_freqs.keys():
				self.class_token_freqs[label] = {}

			if label not in self.label_freqs.keys():
				self.label_freqs[label] = 1
			else:
				self.label_freqs[label] += 1

			for token in tokens:
				self.total_count += 1

				if token not in self.token_freqs.keys():
					self.token_freqs[token] = 1
				else:
					self.token_freqs[token] += 1

				if token not in self.class_token_freqs[label].keys():
					self.class_token_freqs[label][token] = 1
				else:
					self.class_token_freqs[label][token] += 1
