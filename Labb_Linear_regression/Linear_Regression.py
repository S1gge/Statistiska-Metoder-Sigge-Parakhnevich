import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class Linear_Regression:

    def __init__(self, Y, X):
        self.X = np.column_stack([np.ones(Y.shape[0]), X])
        self.Y = Y

    @property
    def fit (self):
        return np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.Y
    
    @property
    def d (self):        # number of features of the model.
        return self.X.shape[1]-1
    
    @property
    def n (self):       # size of the sample.
        return self.Y.shape[0]
    
    @property
    def SSE (self):     # SSE
        return np.sum(np.square(self.Y-(self.X @ self.fit)))
        
    @property
    def var (self):      # A function or method to calculate the variance.
        return self.SSE/(self.n-self.d-1)
    
    @property
    def st_dev (self):      # A function or method to calculate the standard deviation.
        return np.sqrt(self.var)
    
    @property
    def Syy (self):     # Calculate the total variability
        return (self.n*np.sum(np.square(self.Y)) - np.square(np.sum(self.Y)))/self.n
    
    @property
    def SSR (self):     # Calculate the sum of squares due to the regression 
        return self.Syy - self.SSE
    
    @property
    def Rsq (self):
        return self.SSR/self.Syy