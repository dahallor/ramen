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

def convert_column_continuous():
    pass


if __name__ == '__main__':
    filename = 'ramen-ratings.csv'
    data = load_data(filename)
    print(data[:,0])