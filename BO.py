import numpy as np
import matplotlib.pyplot as plt
import kernel as ker
import objectivefunction as objective
import scipy.optimize

class BayesianOptimization():

        def __init__(self,Data,interval,K = np.array([]),f = np.array([]),  kerneltype = 'RBF', acquisitiontype = 'LCB', iteration = 0):

            self.Data = Data
            self.kerneltype = kerneltype
            self.acquisitiontype = acquisitiontype
            self.iteration = 0
            self.interval = interval
            self.K = K
            self.f = f
            
        def getK(self):
        
            n = len(self.Data)
            self.K =  np.zeros([n,n])
            for i in range(n):
                for j in range(n):   
                    self.K[i][j] = ker.kernel(self.Data[i][1],self.Data[j][1])

        def getf(self):

            n = len(self.Data)
            self.f =  np.zeros(n)
            for i in range(n):
                self.f[i] =  self.Data[i][0]
            
        def getk(self,x):

            n = len(self.Data)
            k =  np.zeros(n)
            for i in range(n):
                k[i] = ker.kernel(x,self.Data[i][1])
            return k
            
        def getgaussian(self,x):
            
            k = self.getk(x)
            mu = np.dot(k.T,np.dot(np.linalg.inv(self.K),self.f))
            var = ker.kernel(x,x) - np.dot(k.T,np.dot(np.linalg.inv(self.K),k))
            return (mu,var)

        def acquisition(self,x):
    
            kappa = 1        
            (mu,var) = self.getgaussian(x)        
            LCB = mu - kappa*var**0.5
            return LCB
        
        def getnext(self):
        
            bounds = [(self.interval[0],self.interval[1])]
            res = scipy.optimize.minimize(self.acquisition,0,method = 'TNC', bounds = bounds)
            y = objective.objectivefunction(float(res.x))
            self.Data = self.Data + [(y,float(res.x))]
            plt.figure(self.iteration)
            plt.plot(float(res.x),res.fun,'y^')
            self.iteration += 1
        
        def plotGP(self):
        
            x = np.arange(self.interval[0], self.interval[1], 0.01)
            mean = np.zeros(len(x))
            UCB = np.zeros(len(x))
            LCB = np.zeros(len(x))
            acqfun = np.zeros(len(x))
            for i in range(len(x)):
                (mu , var) = self.getgaussian(x[i])
                mean[i] = mu
                sigma = var**0.5
                UCB[i] = mu + 2*sigma
                LCB[i] = mu - 2*sigma
                acqfun[i] = mu - sigma
                
            plt.figure(self.iteration)
            [real,GPmean,Upper,Lower,acq] = plt.plot(x,objective.objectivefunction(x),'r--',x,mean,'b-',x,UCB,'g-',x,LCB,'g-',x,acqfun,'y-')
            plt.legend([real,GPmean,Upper,Lower,acq], ["$Real$","$Mean$","$UCB$","$LCB$","$Acquisition Function$"])
            plt.xlabel('$x$')
            plt.ylabel('$f(x)$')
            plt.show(block=False)
