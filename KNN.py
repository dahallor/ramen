import numpy as np
from preprocess_data import *
from scipy.stats import mode

class KNN:

    #classes - list of class labels [0, 1, 2, 3, 4, 5]
    def __init__(self, k, X_train, Y_train, X_valid, Y_valid, classes):
        self.k = k
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.validate()
    
    def validate(self):
        accuracy = 0
        for i in range(len(self.X_valid)):
            x_valid = self.X_valid[i]
            y_valid = self.Y_valid[i]
            distances = self.calc_distances(x_valid)
            k_nearest_indices = self.get_k_nearest(distances)
            y_pred = self.get_mean(k_nearest_indices)
            if y_pred == y_valid:
                accuracy += 1
        accuracy = accuracy / len(self.X_valid)
        return accuracy

    #return indices
    def get_k_nearest(self, distances):
        k_nearest_indices = distances.argsort()[:self.k]
        return k_nearest_indices

    #how to break ties?
    #I changed it from mode to mean
    def get_mean(self, k_nearest_indices):
        if self.k == 1:
            return self.Y_train[k_nearest_indices[0]]
        else:
            sum = 0
            for i in k_nearest_indices:
                y = self.Y_train[i]
                sum += y
            mean = sum / self.k
            #round up or round down??
            return mean

    def calc_distances(self, x_valid):
        distances = []
        for x_train in self.X_train:
            distance = self.calc_distance(x_train, x_valid)
            distances.append(distance)
        distances = np.array(distances)
        return distances

    #squared euclidean distance
    def calc_distance(self, x_train, x_valid):
        subtracted = x_train - x_valid
        squared = subtracted * subtracted
        distance = np.sum(squared)
        return distance

def main(filename):
    #preprocess data
    k = 5
    classes = [0, 1, 2, 3, 4, 5]
    #accuracy = KNN(k, X_train, Y_train, X_valid, Y_valid, classes)



if __name__ == '__main__':
    filename = "ramen-ratings.csv"
    main(filename)
