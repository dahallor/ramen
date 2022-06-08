import numpy as np
import sys
import matplotlib.pyplot as plt


def MAPE(Y, Y_hat):
    N = len(Y)
    mape = 0
    for i in range(len(Y)):
        if Y[i] == 0:
            #N = N - 1
            continue
        else:
            mape += abs((Y[i] - Y_hat[i])/Y[i])
    #print(len(np.absolute((Y - Y_hat)/Y)))
    #print(np.sum(np.absolute((Y - Y_hat)/Y)))
    mape = (1/N) * mape
    #print(N)
    #print(mape)
    return mape

def graph(x, y, title, x_label, y_label, figure_name):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(figure_name, format="png")