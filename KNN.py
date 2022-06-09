import numpy as np
from preprocess_data import *
from valid_statistics import *
from scipy.stats import mode

class KNN:

    def __init__(self, X_train, Y_train, X_valid, Y_valid):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
    
    def validate(self, k):
        Y_pred = []
        for i in range(len(self.X_valid)):
            #if i%10 == 0:
                #print(f"{i/len(self.X_valid)}%")
            x_valid = self.X_valid[i]
            y_valid = self.Y_valid[i]
            distances = self.calc_distances(x_valid)
            k_nearest_indices = self.get_k_nearest(distances, k)
            y_pred = self.get_mean(k_nearest_indices, k)
            Y_pred.append(y_pred)
            #print("y_valid", y_valid, "y_pred", y_pred)
        mape = MAPE(self.Y_valid, Y_pred)
        return mape

    #return indices
    def get_k_nearest(self, distances, k):
        k_nearest_indices = distances.argsort()[:k]
        return k_nearest_indices

    def get_mean(self, k_nearest_indices, k):
        if k == 1:
            return self.Y_train[k_nearest_indices[0]]
        else:
            sum = 0
            for i in k_nearest_indices:
                y = self.Y_train[i]
                sum += y
            mean = sum / k
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


if __name__ == '__main__':
    #filename = "ramen-ratings.csv"

    X_train = np.array([[0.1, 0.2, 0.4],
    [0.5, 0.2, 0.3]])
    Y_train = np.array([1, 2])

    X_valid = np.array([[0.5, 0.3, 0.2]])
    Y_valid = np.array([2])

    knn = KNN(X_train, Y_train, X_valid, Y_valid)
    knn.validate(1)
    knn.validate(2)
