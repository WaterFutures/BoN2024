from torch.utils.data import Dataset
from data_loader import load_characteristics, load_calendar
import numpy as np
import torch
from tree import map_structure

class BoNDataset(Dataset):
    def __init__(self, demand_data, climate_data, sequence_len, no_overlap=False, 
                 encoder=None, normalization=None, device='cpu'):
        self.demands = demand_data.astype(np.float32)
        self.climate = climate_data.astype(np.float32)
        self.valid_mask = ~self.demands.isna()
        self.sequence_len = sequence_len
        self.no_overlap = no_overlap
        self.dma_characteristics = load_characteristics()
        self.time = self.compute_time_features(device)
        self.encoder = encoder
        
        if normalization == 'standardize':
            self.standardize_demands()
            self.denormalize = self.unstandardize_demands
        elif normalization == 'normalize':
            self.normalize_demands()
            self.denormalize = self.unnormalize_demands
        else:
            assert normalization is None, 'normalization method not implemented.'
        
        # TODO: Temporarily set nans to zero
        self.demands = self.demands.fillna(0)
        
        # convert to [N, C, L] tensors, where L is sequence length, C is channels
        self.demands = torch.tensor(self.demands.to_numpy().T).to(device)
        self.climate = torch.tensor(self.climate.to_numpy().T).to(device)
        self.valid_mask = torch.tensor(self.valid_mask.to_numpy().T).to(device)

    def to(self, device):
        self.demands = self.demands.to(device)
        self.climate = self.climate.to(device)
        self.valid_mask = self.valid_mask.to(device)
        self.time = map_structure(lambda t: t.to(device), self.time)
        
    def standardize_demands(self):
        self.demand_mean = self.demands.mean(0)
        self.demand_std = self.demands.std(0)
        self.demands = (self.demands - self.demand_mean) / self.demand_std
        
    def unstandardize_demands(self, demands):
        if not (hasattr(self, 'demand_mean') and hasattr(self, 'demand_std')):
            raise Exception(
                'Cannot unstandardize dataset that has not been standardized before.'
            )
        return (demands * self.demand_std) + self.demand_mean   
     
    def normalize_demands(self):
        self.demand_min = self.demands.min(0)
        self.demand_max = self.demands.max(0)
        self.demands = (
            (self.demands - self.demand_min) / 
            (self.demand_max - self.demand_min)
        )
    
    def unnormalize_demands(self, demands):
        if not (hasattr(self, 'demand_min') and hasattr(self, 'demand_max')):
            raise Exception(
                'Cannot unnormalize dataset that has not been normalized before.'
            )
        return demands * (self.demand_max - self.demand_min) + self.demand_min

    def compute_time_features(self, device):
        self.calendar = load_calendar()
        # reduce calendar to data dates
        self.calendar = self.calendar.loc[self.demands.index.date]
        time_features = torch.tensor(np.stack([
            self.demands.index.weekday.to_numpy(),
            self.demands.index.month.to_numpy(),
            self.demands.index.day.to_numpy(),
            self.demands.index.year.to_numpy(),
            self.demands.index.hour.to_numpy(),
            self.calendar.Holiday
        ]))
        time_fields = ('weekday', 'month', 'day', 'year', 'hour', 'holiday')
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
        time = map_structure(lambda ts: ts[idxs], self.time)
        
        sample = {
            'time' : time, 
            'demand' : demand_seq, 
            'climate' : climate_seq
        }

        known_features = None
        
        if self.encoder is not None:
            known_features = self.encoder(sample)

        return {
            'x' : demand_seq,
            'known_features' : known_features,
            'valid_mask' : valid_mask
        }