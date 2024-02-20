import os
import pandas as pd
import numpy as np

DMAS_NAMES = ['DMA_A', 'DMA_B', 'DMA_C', 'DMA_D', 'DMA_E', 'DMA_F', 'DMA_G', 'DMA_H', 'DMA_I', 'DMA_J']
WEATHER_FEATURES = ['Rain', 'Temperature', 'Humidity', 'Windspeed']
WEEK_LEN = 24 * 7

### Data loading helpers
def adjust_summer_time(df):
    days_missing_hour = ['2021-03-28', '2022-03-27', '2023-03-26']

    # Copy 1AM and 3AM data to 2AM for days missing 2AM
    for day in days_missing_hour:
        df = pd.concat([df, df.loc[f'{day} 01:00:00':f'{day} 03:00:00']
                    .reset_index()
                    .assign(Date=pd.to_datetime(f'{day} 02:00:00'))
                    .set_index('Date')]) \
                        .sort_index()

    # Average 2AM values for days with duplicates
    return df.groupby('Date').mean().sort_index()

def adjust_first_monday(df):
    # Make Data start at first monday
    return df.loc['2021-01-04':]

def load_demand_weather(iter_idx=1):
    data_folder = os.getenv('BON2024_DATA_FOLDER')
    if data_folder is None:
        data_folder = 'data'

    # Load the excel file
    rawdata = pd.read_excel(os.path.join(data_folder, 'original', 'InflowData_'+str(iter_idx)+'.xlsx') )

    # Make the first column to datetime format
    rawdata.iloc[:,0] = pd.to_datetime(rawdata.iloc[:,0], format='%d/%m/%Y %H:%M')
    rawdata = rawdata.rename(columns={rawdata.columns[0]: 'Date'})

    demand = rawdata
    demand.set_index('Date', inplace=True) # Make the Date column the index of the dataframe

    # Rename the columns from DMA_A to DMA_J
    demand.columns = DMAS_NAMES

    # Load weather data
    rawdata = pd.read_excel(os.path.join(data_folder, 'original', 'WeatherData_'+str(iter_idx)+'.xlsx') )

    #Same stuff for weather data
    rawdata.iloc[:,0] = pd.to_datetime(rawdata.iloc[:,0], format='%d/%m/%Y %H:%M')
    rawdata = rawdata.rename(columns={rawdata.columns[0]: 'Date'})
    weather = rawdata

    weather.set_index('Date', inplace=True)
    weather.columns = WEATHER_FEATURES

    return demand, weather

def load_data(iter_idx=1):
    demand, weather = load_demand_weather(iter_idx)

    # Adjust data
    demand = adjust_summer_time(demand)
    weather = adjust_summer_time(weather)
    demand = adjust_first_monday(demand)
    weather = adjust_first_monday(weather)

    assert demand.shape[0] == weather.shape[0]-WEEK_LEN
    assert demand.shape[1] == len(DMAS_NAMES)
    assert weather.shape[1] == len(WEATHER_FEATURES)
    assert demand.shape[0] % WEEK_LEN == 0 # Assert I have an integer number of weeks

    return demand, weather
