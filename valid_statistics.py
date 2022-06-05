import numpy as np

def MAPE(Y, Y_hat):
	N = len(Y)
	mape = (1/N) * np.sum(np.absolute((Y - Y_hat)/Y))
	return mape