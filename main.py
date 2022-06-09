from preprocess_data import *
from valid_statistics import MAPE, graph, print_stats
from Regression import LinearRegression
from KNN import KNN
from NaiveBayes import NaiveBayes
import pdb


def main():
    #Preprocess Ramen Dataset
    filename = 'ramen-ratings.csv'
    data = load_data(filename)
    
    X, Y = split_X_Y(data)
    continuous_X = convert_X_to_continuous(X)
    
    #classes 0, 1, 2, 3, 4, 5
    discrete_Y = rounded_Y(Y)
    
    discrete_X_train, discrete_X_valid = split_train_valid(X)
    cont_X_train, cont_X_valid = split_train_valid(continuous_X)
    cont_Y_train, cont_Y_valid = split_train_valid(Y)
    discrete_Y_train, discrete_Y_valid = split_train_valid(discrete_Y)
    
    cont_X_train_mean, cont_X_train_std = calc_mean_std(cont_X_train)
    cont_X_train = z_score(cont_X_train, cont_X_train_mean, cont_X_train_std)
    cont_X_valid = z_score(cont_X_valid, cont_X_train_mean, cont_X_train_std)

    cont_X_train_bias = add_dummy(cont_X_train)
    cont_X_valid_bias = add_dummy(cont_X_valid)

    #NOTE: I did not shuffle the data, not sure if I should have...
    
    run_lr = False
    run_knn = False
    run_naive = True

    if run_lr:
        #Linear Regression
        lr = LinearRegression(cont_X_train_bias, cont_Y_train, cont_X_valid_bias, cont_Y_valid)
        lr.direct_solution()
        lr.gradient_descent()

    if run_knn:
        #KNN
        knn = KNN(cont_X_train, cont_Y_train, cont_X_valid, cont_Y_valid)
        validate_and_graph_knn(knn)

    if run_naive:
        #NaiveBayes
        nb = NaiveBayes(discrete_X_train, discrete_Y_train, discrete_X_valid, discrete_Y_valid)
        nb.train()
        train_preds = nb.get_preds(nb.discrete_X_train)
        print(f"Train MAPE: {MAPE(nb.discrete_Y_train, train_preds)}")
        valid_preds = nb.get_preds(nb.discrete_X_valid)
        print(f"Validation MAPE: {MAPE(nb.discrete_Y_valid, valid_preds)}")
    #NaiveBayes
    #nb = NaiveBayes(discrete_X_train, discrete_Y_train, discrete_X_valid, discrete_Y_valid)
    #TODO: call NB methods
    
def validate_and_graph_knn(knn):
    ks = []
    mapes = []
    print("k =", 1)
    mape = knn.validate(1)
    ks.append(1)
    mapes.append(mape)
    print(mape)
    for k in range(100, 1800, 100):
        print("k =", k)
        mape = knn.validate(k)
        print(mape)
        ks.append(k)
        mapes.append(mape)
    graph(ks, mapes, "MAPEs of KNN models", "K", "MAPE", "knn.png")

if __name__ == '__main__':
    main()