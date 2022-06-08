from preprocess_data import *
import pdb
#Direct solution or regression?

class LinearRegression():

    def __init__(self, X_train, Y_train, X_valid, Y_valid):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
    
    def direct_solution(self):
        X_train_T = np.transpose(self.X_train)
        w = np.linalg.inv(X_train_T @ self.X_train) @ X_train_T @ self.Y_train
    
    def gradient_descent():
        #TODO
        pass


def main(filename):
    #TODO you can use this for testing
    pass


if __name__ == '__main__':
    filename = "ramen-ratings.csv"
    main(filename)