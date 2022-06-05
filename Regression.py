
#Direct solution or regression?

class LinearRegression():

    def __init__(self, X_train, Y_train, X_valid, Y_valid):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
    
    def direct_solution(self):
        pass


def main(filename):
    #preprocess data
    #need to add bias to X
    #Y should be continuous values 
    #accuracy = KNN(k, X_train, Y_train, X_valid, Y_valid, classes)



if __name__ == '__main__':
    filename = "ramen-ratings.csv"
    main(filename)