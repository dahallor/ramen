from preprocess_data import *
#Direct solution or regression?

class LinearRegression():

    def __init__(self, X_train, Y_train, X_valid, Y_valid):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
    
    def direct_solution(self):
        self.X_train_T = np.transpose(self.X_train)
        w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train_bias_T, X_train_bias)), X_train_bias_T), Y_train)


def main(filename):
    #preprocess data
    #need to add bias to X
    X_train_bias = add_dummy(X_train)
    X_valid_bias = add_dummy(X_valid)
    #Y should be continuous values 
    LinearRegression(X_train_bias, Y_train, X_valid_bias, Y_valid)



if __name__ == '__main__':
    filename = "ramen-ratings.csv"
    main(filename)