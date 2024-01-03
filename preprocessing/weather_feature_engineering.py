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
