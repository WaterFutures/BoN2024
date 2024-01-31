from torch.utils.data import Dataset, DataLoader
from .processors import FeatureEncoder
from Utils.data_loader import load_characteristics, load_calendar
import numpy as np
import torch
from tree import map_structure
from datetime import timedelta
import pandas as pd

class BoNDataset(Dataset):
    def __init__(self, demand_data, climate_data, sequence_len, config):
        encoders = config.get('encoders', [])
        self.device = config['device']
        self.demands_df = demand_data.astype(np.float32)
        self.climate_df = climate_data.astype(np.float32)
        self.valid_mask = ~self.demands_df.isna()
        self.sequence_len = sequence_len
        self.dma_characteristics = load_characteristics()
        self.time = self.compute_time_features()
        self.encoder = FeatureEncoder(*encoders)
        
        # Set nans to zeros for now
        self.demands_df = self.demands_df.fillna(0)
        # Set nans of rain data to zeros for now
        rain_data = self.climate_df['Rain'].copy().fillna(0)
        # Interpolate windspeed, humidity and temerature for now
        self.climate_df = self.climate_df.interpolate('linear').bfill().fillna(0)
        self.climate_df['Rain'] = rain_data
        
        self.known_features_list = self.encoder({
            'time' : self.time,
            'demand' : self.demands_df,
            'climate' : self.climate_df
        }, concat=False)
        self.known_features = np.concatenate(self.known_features_list, axis=-1)
        
        # convert to [N, C, L] tensors, where L is sequence length, C is channels
        self.demands = torch.tensor(self.demands_df.to_numpy().T)
        self.climate = torch.tensor(self.climate_df.to_numpy().T)
        self.valid_mask = torch.tensor(self.valid_mask.to_numpy().T)
        self.known_features = torch.tensor(self.known_features.T)
        self.time = map_structure(lambda t: torch.tensor(t), self.time)

    @classmethod
    def from_dataframe(cls, dataframe, sequence_len, config):
        demand_data = dataframe[[ f'DMA_{chr(65 + i)}' for i in range(10) ]]
        climate_data = dataframe[['Rain', 'Temperature', 'Humidity', 'Windspeed']]
        return cls(demand_data, climate_data, sequence_len, config)

    def to(self, device):
        self.device = device
        self.demands = self.demands.to(device)
        self.climate = self.climate.to(device)
        self.valid_mask = self.valid_mask.to(device)
        self.known_features = self.known_features.to(device)
        if hasattr(self, 'week_avg'):
            self.week_avg = self.week_avg.to(device)
        self.time = map_structure(lambda t: t.to(device), self.time)

    def compute_week_avg(self):
        group = [
            self.demands_df.index.weekday, 
            self.demands_df.index.hour
        ]
        week_mean = self.demands_df.groupby(group).mean()
        return week_mean

    def set_week_avg(self, week_avg):
        # extend the week avg by a month, test data goes beyond training date
        idx_ = self.demands_df.index.copy()
        idx_ext = [ idx_[-1] + timedelta(hours=i) for i in range(4*7*24) ]
        idx_.append(pd.Index(idx_ext))
        group = [ idx_.weekday, idx_.hour ]
        self.week_avg = torch.tensor(week_avg.loc[zip(*group)].to_numpy().T)
        self.week_avg = self.week_avg.to(self.device)

    def compute_time_features(self):
        self.calendar = load_calendar()
        # reduce calendar to data dates
        calendar = self.calendar.loc[self.demands_df.index.date]
        time_features = np.stack([
            self.demands_df.index.weekday.to_numpy(),
            self.demands_df.index.month.to_numpy(),
            self.demands_df.index.day.to_numpy(),
            self.demands_df.index.year.to_numpy(),
            self.demands_df.index.hour.to_numpy(),
            self.demands_df.index.days_in_month.to_numpy(),
            calendar.Holiday
        ], dtype=np.float32)
        time_fields = ('weekday', 'month', 'day', 'year', 'hour', 'days_in_month', 'holiday')
        return dict(zip(time_fields, time_features))
    
    @property
    def num_features(self):
        return self.__getitem__(0).shape[-2]

    def __len__(self):
        return len(self.demands[0]) - self.sequence_len + 1

    def __getitem__(self, idx):
        idxs = slice(idx, idx + self.sequence_len)
        demand_seq = self.demands[:, idxs]
        climate_seq = self.climate[:, idxs]
        valid_mask = self.valid_mask[:, idxs]
        known_features = self.known_features[:, idxs]
        week_avg = self.week_avg[:, idxs]

        return {
            'x' : demand_seq,
            'known_features' : known_features,
            'valid_mask' : valid_mask,
            # 'bias_featues' : bias_featues
            'residuals' : week_avg
        }
        
def load_dataframe(demands, climate, target_len, config, start_idx=None, stop_idx=None, shuffle=False):
    index = demands.index[slice(start_idx, stop_idx)]
    demands = demands.loc[index]
    climate = climate.loc[index]
    seq_len = target_len + config['seq_seed_len']
    dataset = BoNDataset(demands, climate, seq_len, config)
    dataset.to(config['device'])
    data_loader = DataLoader(dataset, config['batch_size'], shuffle=shuffle)
    return data_loader