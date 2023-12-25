import numpy as np
import pandas as pd


class Preprocessing:

    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X
    


class Logarithm(Preprocessing):

    def __init__(self):
        super().__init__()

    def transform(self, X):
        return np.log(X)
    
    def inverse_transform(self, X):
        return np.exp(X)
    




