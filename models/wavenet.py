import os
import warnings
import numpy as np
import torch
import random
import logging
from models.base import Model
from .wavenet_impl.torch_dataset import BoNDataset
from .wavenet_impl.processors import FeatureEncoder, Preprocessor, ENCODERS
from .wavenet_impl.metrics import MulticlassMetrics
from .wavenet_impl.wavenet import WaveNet
import yaml
import os
import torch
import pandas as pd
from preprocessing.simple_transforms import Logarithm

class WaveNetModel(Model):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def fit(self, demand_train, weather_train):
        self.model = WaveNet(cfg).to(cfg['device'])
        self.model.fit(demand_train, weather_train)
    
    def forecast(self, demand_data, weather_data):
        return self.model.forecast(demand_data, weather_data)



CONFIG = './models/wavenet_impl/config.yml'

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)
    
cfg['device'] = 'cuda'
cfg['encoders'] = [ k for k in ENCODERS if cfg.pop(k) ]

class WaveNet_prepare_test_dfs:

    def transform(self, demand_train, weather_train, weather_test):
        demand_nans = pd.DataFrame(columns=demand_train.columns, data=np.nan, index=weather_test.index)
        demands_test = pd.concat([demand_train, demand_nans], axis=0)
        weather_test = pd.concat([weather_train, weather_test], axis=0)
        return demands_test, weather_test
 