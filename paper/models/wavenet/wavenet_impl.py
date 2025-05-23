import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../..")
from models.base import Model
from models.wavenet_impl.processors import ENCODERS
from models.wavenet_impl.wavenet import WaveNet

class WaveNetModel(Model):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config['encoders'] = [ k for k in ENCODERS if self.config.pop(k) ]
    
    def fit(self, demand_train, weather_train):
        self.model = WaveNet(self.config).to(self.config['device'])
        self.model.fit(demand_train, weather_train)
    
    def forecast(self, demand_data, weather_data):
        return self.model.forecast(demand_data, weather_data)
    