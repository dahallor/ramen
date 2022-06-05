from preprocess_data import *
from Regression import LinearRegression
from KNN import KNN
from NaiveBayes import NaiveBayes

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
    lr = LinearRegression(cont_X_train_bias, Y_train, cont_X_valid_bias, Y_valid)
    #TODO: call LR methods

    #KNN
    knn = KNN(cont_X_train, Y_train, cont_X_valid, Y_valid)
    #TODO: call KNN methods

    #NaiveBayes
    nb = NaiveBayes(discrete_X_train, Y_train, discrete_X_valid, Y_valid)
    #TODO: call NB methods


if __name__ == '__main__':
    main()