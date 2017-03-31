import numpy as np

def objectivefunction(x):

    return (x/10)**2 + np.sin(x)
    #return -(-x**2 + 3*np.sin(x)-4*np.exp(x/10))+10*np.exp(-x**2)
