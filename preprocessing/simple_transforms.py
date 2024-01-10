from preprocessing.base import Preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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