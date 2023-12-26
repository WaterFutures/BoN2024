import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
    


    




