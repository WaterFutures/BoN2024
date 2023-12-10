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

class AutoregressiveTransform:

    def __init__(self, seed_len, target_length):
        # seed length determines the number of timesteps that the model is conditioned on
        self.seed_len = seed_len

    def __call__(self, sample):
        demand = sample['demand']
        # compute all additional encodings (e.g. climate, time, ...)
        encs = torch.cat([ fn(sample) for fn in self.encoders ], dim=-2)
        # concat additional encodings to demand signal
        processed = torch.cat([demand, encs], dim=-2)
        predict_values_mask = torch.cat([
            torch.ones_like(demand, dtype=torch.bool), 
            torch.zeros_like(encs, dtype=torch.bool)], 
        dim=-2)
        return processed, predict_values_mask