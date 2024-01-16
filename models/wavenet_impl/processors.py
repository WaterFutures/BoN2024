import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from metpy.units import units
from metpy import calc
from itertools import count
from functools import partial

TEMP_UNITS = '°C'
HUMIDITY_UNITS = '%'
WINDSPEED_UNITS = 'km/h'

# scaling values
SCALING_VALUES = {
    'temperature' : 1./30,
    'windspeed'   : 1./50,
    'humidity'    : 1./100,
    'rain'    : 1./20
}

# --- encoders ---
def angular_hour_encoder(data):
    hour = data['time']['hour']
    hour = (hour / 24) * 2 * np.pi
    return np.stack((np.sin(hour), np.cos(hour)), axis=-1)

def one_hot_weekday_encoder(data):
    weekday = data['time']['weekday'].astype(int)
    weekday = np.eye(7, dtype=np.float32)[weekday]
    return weekday

def angular_month_encoder(data):
    time = data['time'].copy()
    month = time['month']
    day = time['day'] / time['days_in_month']
    month = ((month + day) / 12) * 2 * np.pi
    return np.stack((np.sin(month), np.cos(month)), axis=-1)

def angular_discrete_month_encoder(data):
    time = data['time'].copy()
    month = time['month']
    month = (month / 12) * 2 * np.pi
    return np.stack((np.sin(month), np.cos(month)), axis=-1)

def business_days(data, scale=1/6.):
    sundays = data['time']['weekday'] == 6
    public_holidays = data['time']['holiday'] == 1
    non_business_days = np.logical_or(sundays, public_holidays)#.astype(int)
    #non_business_days = pd.DataFrame(non_business_days, index=sundays.index)

    since_offday, counter = [], count()
    
    for day_off_since in non_business_days:
        if day_off_since:
            counter = count()
        since_offday.append(next(counter) / 24)

    to_offday, counter = [], count()

    for day_off_to in reversed(non_business_days):
        if day_off_to:
            counter = count()
        to_offday.append(next(counter) / 24)

    return scale * np.stack((to_offday[::-1], since_offday), axis=-1).astype(np.float32)


def apparent_temperature(data, scale=1./30):
    climate = data['climate'].copy()
    real_feel = calc.apparent_temperature(
        climate['Temperature'].values * units(TEMP_UNITS),
        climate['Humidity'].values * units(HUMIDITY_UNITS),
        climate['Windspeed'].values * units(WINDSPEED_UNITS),
        mask_undefined=False
    )
    return scale * np.array(real_feel.m, dtype=np.float32)[:, None]

def dewpoint(data, scale=1./30):
    climate = data['climate'].copy()
    dewpoint = calc.dewpoint_from_relative_humidity(
        climate['Temperature'].values * units(TEMP_UNITS),
        climate['Humidity'].values * units(HUMIDITY_UNITS)
    )
    return scale * np.array(dewpoint.m, dtype=np.float32)[:, None]

def discounted_accumulated_rain_volume(data, discount_factor=0.99, scale=1./50.):
    last_rain_vol = data['climate']['Rain'].copy()
    lasti = last_rain_vol.index[0]

    for i, s in last_rain_vol.items():
        last_rain_vol.loc[i] += last_rain_vol.loc[lasti] * discount_factor
        lasti = i
    
    return scale * np.array(last_rain_vol, dtype=np.float32)[:, None]

def average_week_rain(data, scale=2.):
    climate = data['climate'].copy()
    mean_rain_week = climate['Rain'].rolling(7*24, 24).mean().bfill().fillna(0)
    return scale * np.array(mean_rain_week, dtype=np.float32)[:, None]

def days_since_last_rain(data, scale=1./14):
    rain_rate = data['climate']['Rain'].copy()
    counter = count()
    days_since_rain = []
    for is_raining in (rain_rate > 0):
        if is_raining:
            counter = count()
        # hourly measurements, so divide by 24 to convert hours to days
        days_since_rain.append(next(counter) / 24.)
    return scale * np.array(days_since_rain, dtype=np.float32)[:, None]

def climate_attribute(data, attribute=None):
    climate = data['climate'][attribute.capitalize()].copy()
    scale = SCALING_VALUES[attribute.lower()]
    return scale * np.array(climate, dtype=np.float32)[:, None]

ENCODERS = {
    'angular_hour_encoder' : angular_hour_encoder,
    'one_hot_weekday_encoder' : one_hot_weekday_encoder,
#    'angular_month_encoder' : angular_month_encoder,
    'angular_discrete_month_encoder' : angular_discrete_month_encoder,
    'business_days' : business_days,
    'apparent_temperature' : apparent_temperature,
    'dewpoint' : dewpoint,
    'discounted_accumulated_rain_volume' : discounted_accumulated_rain_volume,
    'average_week_rain' : average_week_rain,
    'days_since_last_rain' : days_since_last_rain,
    'temperature' : partial(climate_attribute, attribute='temperature'),
    'windspeed' : partial(climate_attribute, attribute='windspeed'),
    'humidity' : partial(climate_attribute, attribute='humidity'),
    'rain' : partial(climate_attribute, attribute='rain'),
}

def build_encoders(encoders):
    return [ ENCODERS[e] for e in encoders ]

class FeatureEncoder:

    def __init__(self, *encoders, **kwargs):
        assert len(encoders) == len(np.unique(encoders))
        self.encoders = build_encoders(encoders)

    def __call__(self, sample, concat=True):
        # compute all additional encodings (e.g. climate, time, ...)
        encs = [ fn(sample) for fn in self.encoders ]#, axis=-1)
        if concat:
            return np.concatenate(encs, axis=-1)
        return encs

class TorchStandardScaler(torch.nn.Module):

    def to(self, device):
        self.mean.to(device)
        self.std.to(device)

    def fit(self, x):
        self.mean = torch.nn.Parameter(x.mean(-1, keepdim=True), requires_grad=False)
        self.std = torch.nn.Parameter(x.std(-1, unbiased=False, keepdim=True), requires_grad=False)

    def transform(self, x, eps=1e-8):
        x = x - self.mean
        x /= (self.std + eps)
        return x # = (x - mean) / std

    def inverse_transform(self, x, eps=1e-8):
        x = x * (self.std + eps)
        x += self.mean
        return x # = x * std + mean

class TorchMinMaxScaler(torch.nn.Module):

    def to(self, device):
        self.max.to(device)
        self.min.to(device)

    def fit(self, x):
        self.max = torch.nn.Parameter(x.max(-1, keepdim=True).values, requires_grad=False)
        self.min = torch.nn.Parameter(x.min(-1, keepdim=True).values, requires_grad=False)

    def transform(self, x, eps=1e-8):
        x = x - self.min
        x /= (self.max - self.min + eps)
        return x # = (x - min) / (max - min)

    def inverse_transform(self, x, eps=1e-8):
        x = x * (self.max - self.min + eps)
        x += self.min
        return x # = x * (max - min) + min

class TorchQuantileScaler(TorchMinMaxScaler):

    def __init__(self, quantile):
        super().__init__()
        self.q = quantile

    def fit(self, x):
        self.max = torch.nn.Parameter(torch.quantile(x, self.q, dim=-1, keepdim=True), requires_grad=False)
        self.min = torch.nn.Parameter(x.min(-1, keepdim=True).values, requires_grad=False)

class Preprocessor(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        normalization = config['normalization']

        if normalization == 'standardize':
            self.normalizer = TorchStandardScaler()
        elif normalization == 'normalize':
            self.normalizer = TorchMinMaxScaler()
        elif normalization.startswith('quantile'):
            _, quantile = normalization.split('-')
            self.normalizer = TorchQuantileScaler(float(quantile))
        else:
            assert normalization is None, 'normalization method not implemented.'

        self.apply_log_scaling = config['apply_log_scaling']
        assert not (self.apply_log_scaling and normalization == 'standardize'), \
            'log scaling is not suitable when standardizing'

    def fit(self, data):
        self.normalizer.fit(data)

    # TODO: Doublecheck that this works as inteded
    def inv(self, data):
        data = self.normalizer.inverse_transform(data)
        # if self.apply_log_scaling:
        #     data[data > 0] = torch.exp(data[data > 0]) - 1# * 2. - 2) - torch.exp(torch.tensor(-2))
        return data
    
    def transform(self, data):
        # if self.apply_log_scaling:
        #     data[data > 0] = torch.log(data[data > 0] + 1) #+ 2.) / 2.
        data = self.normalizer.transform(data)
        return data