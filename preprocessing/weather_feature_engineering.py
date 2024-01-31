from preprocessing.base import Preprocessing
from Utils.weather_features import feels_like, wind_chill, heat_index, dew_point, Temp
import pandas as pd
    
class RealFeel(Preprocessing):

    def transform(self, X):
        val = X.apply(lambda x:feels_like(temperature=Temp(x.at['Temperature'], 'c'), humidity=x.at['Humidity'], wind_speed=x.at['Windspeed']).c, axis=1, result_type='reduce')
        val.name = 'real_feel'
        return pd.concat([X, val], axis=1)
    
    
class WindChill(Preprocessing):

    def transform(self, X):
        val = X.apply(lambda x:wind_chill(temperature=Temp(x.at['Temperature'], 'c'), wind_speed=x.at['Windspeed']).c, axis=1, result_type='reduce')
        val.name = 'wind_chill'
        return pd.concat([X, val], axis=1)
    
    
class HeatIndex(Preprocessing):

    def transform(self, X):
        val = X.apply(lambda x:heat_index(temperature=Temp(x.at['Temperature'], 'c'), humidity=x.at['Humidity']).c, axis=1, result_type='reduce')
        val.name = 'heat_index'
        return pd.concat([X, val], axis=1)
    
    
class DewPoint(Preprocessing):

    def transform(self, X):
        val = X.apply(lambda x:dew_point(temperature=Temp(x.at['Temperature'], 'c'), humidity=x.at['Humidity']).c, axis=1, result_type='reduce')
        val.name = 'dew_point'
        return pd.concat([X, val], axis=1)

class DaysSinceRain(Preprocessing):

    def transform(self, X):
        X['temp'] = X['Rain'].eq(0)
        X['days_since_rain'] = (X['temp'] * (X.groupby((X['temp'] != X['temp'].shift()).cumsum()).cumcount() + 1))
        X.drop(columns=['temp'], inplace=True)
        return X
    
"""Cooling and Heating degree days, defined acccording to 
https://www.eia.gov/energyexplained/units-and-calculators/degree-days.php#:~:text=Cooling%20degree%20days%20(CDDs)%20are,two%20days%20is%2033%20CDDs.
"""
class CoolingDegreeDay(Preprocessing):

    def __init__(self, base_temperature=18) -> None:
        self.base_temperature = base_temperature
        
    def transform(self, X):
        # Compute daily mean temperature
        daily_mean_temp = X['Temperature'].groupby(pd.Grouper(freq='D')).mean()
        # CDD is the difference between daily mean temperature and base temperature
        X['cdd'] = daily_mean_temp.apply(lambda x: max(0, x - self.base_temperature)).resample('H').ffill()
        # Smooth out the CDD
        X['cdd'] = X['cdd'].rolling(24*3).mean()

        return X


class HeatingDegreeDay(Preprocessing):

    def __init__(self, base_temperature=18) -> None :
        self.base_temperature = base_temperature
        
    def transform(self, X):
        # Compute daily mean temperature
        daily_mean_temp = X['Temperature'].groupby(pd.Grouper(freq='D')).mean()
        # HDD is the difference between base temperature and daily mean temperature
        X['hdd'] = daily_mean_temp.apply(lambda x: max(0, self.base_temperature - x)).resample('H').ffill()
        # Smooth out the HDD
        X['hdd'] = X['hdd'].rolling(24*3).mean()

        return X

class AmountOfRainDecayed(Preprocessing):

    def __init__(self, discount_factor=0.9):
        super().__init__()
        self.discount_factor = discount_factor

    def transform(self, X):
        last_rain_vol = X['Rain'].copy()
        lasti = last_rain_vol.index[0]

        for i, s in last_rain_vol.items():
            last_rain_vol.loc[i] += last_rain_vol.loc[lasti] * self.discount_factor
            lasti = i
        
        X['AmountOfRain'] = last_rain_vol
        
        return X




    
    