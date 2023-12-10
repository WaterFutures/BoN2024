import torch
import numpy as np

# --- encoders ---
def angular_hour_encoder(sample):
    hour = sample['time']['hour']
    hour = (hour / 24) * 2 * np.pi
    return torch.stack((torch.sin(hour), torch.cos(hour)), dim=-2)

def one_hot_weekday_encoder(sample):
    weekday = sample['time']['weekday']
    weekday = torch.eye(7, device=weekday.device)[:, weekday]
    return weekday

def angular_month_encoder(sample):
    month = sample['time']['month']
    month = (month / 12) * 2 * np.pi
    return torch.stack((torch.sin(month), torch.cos(month)), dim=-2)

ENCODERS = {
    'angular_hour_encoder' : angular_hour_encoder,
    'one_hot_weekday_encoder' : one_hot_weekday_encoder,
    'angular_month_encoder' : angular_month_encoder,
}

def build_encoders(encoders):
    return [ ENCODERS[e] for e in encoders ]

class DataProcessor:

    def __init__(self, *encoders, **kwargs):
        assert len(encoders) == len(np.unique(encoders))
        self.encoders = build_encoders(encoders)

    def __call__(self, sample):
        demand = sample['demand']
        # compute all additional encodings (e.g. climate, time, ...)
        encs = torch.cat([ fn(sample) for fn in self.encoders ], dim=-2)
        return encs