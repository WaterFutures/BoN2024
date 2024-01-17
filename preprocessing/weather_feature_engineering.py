from preprocessing.base import Preprocessing
from Utils.weather_features import feels_like, wind_chill, heat_index, dew_point, Temp
   
    
class RealFeel(Preprocessing):

    def transform(self, X):
        X['real_feel'] = X[['Temperature','Humidity','Windspeed']].apply(lambda x:feels_like(temperature=Temp(x[0], 'c'), humidity=x[1], wind_speed=x[2]).c, axis=1)
        return X
    
    
class WindChill(Preprocessing):

    def transform(self, X):
        X['wind_chill'] = X[['Temperature','Humidity','Windspeed']].apply(lambda x:wind_chill(temperature=Temp(x[0], 'c'), wind_speed=x[2]).c, axis=1)
        return X
    
    
class HeatIndex(Preprocessing):

    def transform(self, X):
        X['heat_index'] = X[['Temperature','Humidity','Windspeed']].apply(lambda x:heat_index(temperature=Temp(x[0], 'c'), humidity=x[1]).c, axis=1)
        return X
    
    
class DewPoint(Preprocessing):

    def transform(self, X):
        X['dew_point'] = X[['Temperature','Humidity','Windspeed']].apply(lambda x:dew_point(temperature=Temp(x[0], 'c'), humidity=x[1]).c, axis=1)
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




    
    