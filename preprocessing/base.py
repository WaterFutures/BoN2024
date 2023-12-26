import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessing:

    def __init__(self):
        pass

    def fit(self, X):
        ''' Not Required, use if you need to save parameters '''
        pass

    def transform(self, X):
        ''' Required for demands and weather '''
        return X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        ''' Required for demands only '''
        return X
    


    




