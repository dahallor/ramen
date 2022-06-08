from preprocess_data import *
import pdb
#Direct solution or regression?

class LinearRegression():

    def __init__(self, X_train, Y_train, X_valid, Y_valid):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        #for plotting effectiveness by epoch
        self.mean_train = []
        self.mean_valid = []
        self.epoch_list = []
        self.epoch = 0

    def _initWeights(self):
        self.W = np.zeros((len(self.X_train[0]), 1), dtype = float)
        for i in range(len(self.X_train[0])):
            rand = np.random.uniform(low=-.0001, high=.0002)
            self.W[i][0] = rand
        self.b = np.random.uniform(low = -.0001, high = .0002)

    def _set_dJdW(self, X, Y, Yhat):
        Xt = X.transpose()
        N = len(X)
        dif = Yhat - Y
        w = np.matmul(Xt, dif)
        self.dJdW = (1/N) * w

    def _set_dJdb(self, Y, Yhat):
        sum = 0
        N = len(Y)
        for i in range(N):
            sum += Yhat[i][0] - Y[i][0]
        self.dJdb = (1/N) * sum

    def _updateWeightsAndBias(self, eta):
        self.W = self.W - eta * self.dJdW
        self.b = self.b - eta * self.dJdb

    def _logLoss(self, y, yhat, input_type):
        epsilon = .0000001
        J = np.zeros((len(y), 1), dtype = 'float')
        for i in range(len(J)):
            J[i][0] = -1 * ((y[i][0] * np.log((yhat[i][0] + epsilon))) + (1 - y[i][0]) * np.log((1 - yhat[i][0] + epsilon)))
        if input_type == "train":
            mean = np.mean(J)
            self.mean_train.append(mean)
        if input_type == "valid":
            mean = np.mean(J)
            self.mean_valid.append(mean)


    def _calcProbability(self, X):
        Yhat = np.zeros((len(X), 1), dtype = 'float')
        for i in range(len(X)):
            exponent = -1 * (np.matmul(X[i], self.W) + self.b)
            Yhat[i] = 1/(1 + np.exp(exponent))
        return Yhat
    
    def direct_solution(self):
        X_train_T = np.transpose(self.X_train)
        w = np.linalg.inv(X_train_T @ self.X_train) @ X_train_T @ self.Y_train
    
    def gradient_descent(self):
        #initalize weight matrix
        self._initWeights()
        eta = .0001
        
        while self.epoch <= 50000:
            pdb.set_trace()
            #Check Conditionals
            if self.epoch % 100 == 0:
                print("Epoch: {}\n".format(self.epoch))
            '''
            if self.epoch > 3:
                change = abs(self.mean_train[self.epoch-1] - self.mean_train[self.epoch-2])
                if change < .0000000001:
                    break
            '''
            
            #Get Yhat        
            trainYhat = self._calcProbability(self.X_train)
            validYhat = self._calcProbability(self.X_valid)

            #Objective Function Evaluation
            self._logLoss(self.Y_train, trainYhat, "train")
            self._logLoss(self.Y_valid, validYhat, "valid")

            #Set Weight and Bias derivatives
            self._set_dJdW(self.X_train, self.Y_train, trainYhat)
            self._set_dJdb(self.Y_train, trainYhat)

            #Update Weights with derivatives
            self._updateWeightsAndBias(eta)

            #Incremenet values
            self.epoch_list.append(self.epoch)
            self.epoch += 1
        
        


def main(filename, data):
    #TODO you can use this for testing
    trans = data.transpose()
    Yt = trans[-1]
    Y = np.array([Yt])
    Y = Y.transpose()
    X = np.delete(data, -1, axis = 1)

    training_size = 40

    trainX = X[:training_size]
    trainY = Y[:training_size]
    validX = X[training_size:]
    validY = Y[training_size:]

    #pdb.set_trace()

    return trainX, validX, trainY, validY
    


if __name__ == '__main__':
    filename = "ramen-ratings.csv"
    test = np.arange(250).reshape((50, 5), order = 'c')
    X_train, X_valid, Y_train, Y_valid = main(filename, test)
    lr = LinearRegression(X_train, Y_train, X_valid, Y_valid)
    lr.gradient_descent()
    