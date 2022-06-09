from preprocess_data import *
from valid_statistics import *
import pdb
from tqdm import tqdm

class LinearRegression():

    def __init__(self, X_train, Y_train, X_valid, Y_valid):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        #for plotting effectiveness by epoch
        self.MAPE_train = []
        self.MAPE_valid = []
        self.epoch_list = []
        self.epoch = 0

#===============================================================================Weights & Biases=================================================================================================================
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


#====================================================================================Alter X & Y============================================================================================
    def _transposeY(self):
        new_Y_train = np.zeros((len(self.Y_train), 1))
        new_Y_valid = np.zeros((len(self.Y_valid), 1))

        for i in range(len(new_Y_train)):
            new_Y_train[i][0] = self.Y_train[i]

        for i in range(len(new_Y_valid)):
            new_Y_valid[i][0] = self.Y_valid[i]

        self.Y_train = new_Y_train
        self.Y_valid = new_Y_valid


    def _remove_dummy_vector(self):
        self.X_train = np.delete(self.X_train, 0, 1)
        self.X_valid = np.delete(self.X_valid, 0, 1)
        




#==============================================================================================Evaluation========================================================================================


    def _calcYhat(self, X):
        Yhat = X.dot(self.W)
        Yhat = Yhat + self.b
        return Yhat

    def _calcYhatDirect(self, X, w):
        Yhat = X.dot(w)
        return Yhat

    def _plotMAPE(self):
        plt.xlabel("Epoch")
        plt.ylabel("MAPE")
        plt.plot(self.epoch_list, self.MAPE_train, label = "Training", color = "blue")
        plt.plot(self.epoch_list, self.MAPE_valid, label = "Validation", color = "orange")
        plt.title(label = "MAPE Vs. Epoch")
        plt.legend()
        plt.show()

#====================================================================================Run Regression=================================================================================================
    def direct_solution(self):
        X_train_T = np.transpose(self.X_train)
        w = np.linalg.inv(X_train_T @ self.X_train) @ X_train_T @ self.Y_train

        Yhat_train = self._calcYhatDirect(self.X_train, w)
        Yhat_valid = self._calcYhatDirect(self.X_valid, w)

        print("Direct:")
        print("{}".format(MAPE(self.Y_train, Yhat_train)))
        print("{}".format(MAPE(self.Y_valid, Yhat_valid)))
        print()

    
    def gradient_descent(self):
        print("executing batch gradient descent")
        #initalize weight matrix
        self._remove_dummy_vector()
        self._transposeY()
        self._initWeights()
        eta = .0001
        
        for i in tqdm(range(15001)):
        #while self.epoch <= 15000:
            #Check Conditionals
            #if self.epoch % 100 == 0:
            #    print("Epoch: {}\n".format(self.epoch))
                #pdb.set_trace()

            
            #Get Yhat        
            trainYhat = self._calcYhat(self.X_train)
            validYhat = self._calcYhat(self.X_valid)


            #Set Weight and Bias derivatives
            self._set_dJdW(self.X_train, self.Y_train, trainYhat)
            self._set_dJdb(self.Y_train, trainYhat)

            #Update Weights with derivatives
            self._updateWeightsAndBias(eta)

            #Incremenet values
            self.epoch_list.append(self.epoch)
            self.epoch += 1

            #Record MAPE value at current Epoch
            train_MAPE_value = MAPE(self.Y_train, trainYhat)
            valid_MAPE_value = MAPE(self.Y_valid, validYhat)
            self.MAPE_train.append(train_MAPE_value)
            self.MAPE_valid.append(valid_MAPE_value)

        print("Gradient Descent:")
        print("Training MAPE: {}".format(MAPE(self.Y_train, trainYhat)))    
        print("Validation MAPE: {}".format(MAPE(self.Y_valid, validYhat)))
        self._plotMAPE()
        
        


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
    