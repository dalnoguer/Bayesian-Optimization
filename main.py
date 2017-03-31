import numpy as np
import objectivefunction as objective
import BO

interval = tuple((-30,30))
initial_samples = 2
iteration_max = 10

# set initial D
D = []
for i in range(initial_samples):

	x = np.random.uniform(interval[0],interval[1])
	y = objective.objectivefunction(x)
	D = D + [(y,x)]

# initialize BO

GP = BO.BayesianOptimization(D,interval)

# iterate to find successive data samples
iteration = 0
while iteration <= iteration_max:
    GP.getK()	
    GP.getf()
    GP.plotGP()
    GP.getnext()
    iteration += 1








