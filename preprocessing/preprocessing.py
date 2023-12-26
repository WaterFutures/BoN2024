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
    


class Logarithm(Preprocessing):

    def transform(self, X):
        return np.log(X)
    
    def inverse_transform(self, X):
        return np.exp(X)
    

class Normalize(Preprocessing):

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)

    def transform(self, X):
        return pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)
    
    def inverse_transform(self, X):
        return pd.DataFrame(self.scaler.inverse_transform(X), index=X.index, columns=X.columns)
    




