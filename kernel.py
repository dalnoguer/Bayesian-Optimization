import numpy as np

# implement different types of kernel functions

def kernel(x1,x2):

	param = 5.0
	ker = np.exp(-np.linalg.norm(x1-x2)**2/param**2)
	return ker

