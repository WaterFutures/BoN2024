import random
import numpy as np

class Model:

    def __init__(self):
        pass

    def fit(self, demands, weather):
        pass

    def forecast(self, demand_test, weather_test):
        pass

    def set_seed(self, seed):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
       
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
