from preprocess_data import *
from valid_statistics import graph
from Regression import LinearRegression
from KNN import KNN
#from NaiveBayes import NaiveBayes

def main():
    #Preprocess Ramen Dataset
    filename = 'ramen-ratings.csv'
    data = load_data(filename)
    
    X, Y = split_X_Y(data)
    continuous_X = convert_X_to_continuous(X)
    
    discrete_X_train, discrete_X_valid = split_train_valid(X)
    cont_X_train, cont_X_valid = split_train_valid(continuous_X)
    Y_train, Y_valid = split_train_valid(Y)
    
    cont_X_train_mean, cont_X_train_std = calc_mean_std(cont_X_train)
    cont_X_train = z_score(cont_X_train, cont_X_train_mean, cont_X_train_std)
    cont_X_valid = z_score(cont_X_valid, cont_X_train_mean, cont_X_train_std)

    cont_X_train_bias = add_dummy(cont_X_train)
    cont_X_valid_bias = add_dummy(cont_X_valid)

    #NOTE: I did not shuffle the data, not sure if I should have...

    #Linear Regression
    #lr = LinearRegression(cont_X_train_bias, Y_train, cont_X_valid_bias, Y_valid)
    #TODO: call LR methods

    #KNN
    print(len(Y_train))
    knn = KNN(cont_X_train, Y_train, cont_X_valid, Y_valid)
    ks = []
    mapes = []
    for k in range(100, 1800, 100):
        print("k =", k)
        mape = knn.validate(k)
        ks.append(k)
        mapes.append(mape)
        print(round(mape, 3))
    graph(ks, mapes, "MAPEs of KNN models", "K", "MAPE", "knn.png")

    #NaiveBayes
    #nb = NaiveBayes(discrete_X_train, Y_train, discrete_X_valid, Y_valid)
    #TODO: call NB methods


if __name__ == '__main__':
    main()