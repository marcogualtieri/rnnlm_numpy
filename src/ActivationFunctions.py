#### ActivationFunctions module

import sys
import numpy as np
from scipy import special as sp_special

FLT_MAX = sys.float_info.max

# sigmoid function
def sigmoid(x): 
	return 1.0/(1.0+np.exp(-x)) #sp_special.expit(x)

# derivative of sigmoid function
def sigmoid_d(y):
	return y*(1.0-y)

# softmax function
def softmax(x):
	exp = np.exp(x)
	return exp / np.sum(exp)

# derivative of softmax function
def softmax_d(y):
	d_y = np.outer(-y,y)
	d_y[np.diag_indices_from(d_y)] += y 
	return d_y