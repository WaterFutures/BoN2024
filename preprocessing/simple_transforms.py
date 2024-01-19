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
    


class SummarizeDays(Preprocessing):
    
    def transform(self, X):
        # print(X.head())
        data_aggregated = pd.DataFrame(columns=X.columns)
        for c in X.columns:
            data = X[c].to_numpy()
            # print(data.shape)
            mean = []
            for i in range(int(len(data)/24)):
                mean.append(np.mean(data[i*24:(i+1)*24]))
            mean = np.hstack(mean)
            # print(mean.shape)
            data_aggregated[c]=mean

        return data_aggregated

class LinearInterpolation(Preprocessing):

    def __init__(self, interpolation_range):
        super().__init__()
        self.interpolation_range = interpolation_range

    def transform(self, X):
        
        header = X.columns
        index = X.index
        # interpolating missing values - issue: limit option interpolates limit values for imputation even if many more are missing
        interpolated = X.interpolate(axis='rows', limit_area='inside', inplace=False).to_numpy()
        
        # compute mask to detmine for one dma whether the values should be imputed or not according to self.interpolation_range
        def compute_mask(input):
            m = np.ones(len(input), dtype=int)

            for i in range(len(input)-self.interpolation_range):
                if np.isnan(input[i:i+self.interpolation_range]).sum() == self.interpolation_range:
                    for j in range(self.interpolation_range):
                        m[i+j] = 0            
            return m
        
        fct = lambda m, v : v if m else np.nan

        res = np.zeros((interpolated.shape))
        X = X.to_numpy()

        for dma in range(X.shape[1]):
            mask = compute_mask(X[:, dma])
            for i in range(len(res)):
                res[i, dma] = fct(mask[i], interpolated[i,dma])

        return pd.DataFrame(res, columns=header, index=index)

