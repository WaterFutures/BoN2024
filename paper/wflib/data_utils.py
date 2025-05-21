import os
import pandas as pd

DMAS_NAMES = ['DMA_A', 'DMA_B', 'DMA_C', 'DMA_D', 'DMA_E', 'DMA_F', 'DMA_G', 'DMA_H', 'DMA_I', 'DMA_J']

def load_characteristics() -> pd.DataFrame:
    dma_characts_json = {
        'name_short':['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], 
        'name_long': DMAS_NAMES,
        'description' : ['Hospital district', 'Residential district in the countryside', 'Residential district in the countryside',
                        'Suburban residential/commercial district', 'Residential/commercial district close to the city centre',
                        'Suburban district including sport facilities and office buildings', 'Residential district close to the city centre',
                        'City centre district', 'Commercial/industrial district close to the port', 'Commercial/industrial district close to the port'],
        'desc_short' : ['Hosp', 'Res cside', 'Res cside', 
                        'Suburb res/com', 'Res/com close',
                        'Suburb sport/off', 'Res close',
                        'City', 'Port', 'Port'],
        'population' : [162, 531, 607, 2094, 7955, 1135, 3180, 2901, 425, 776],
        'h_mean' : [8.4, 9.6, 4.3, 32.9, 78.3, 8.1, 25.1, 20.8, 20.6, 26.4] # L/s/per hour
    }
    dmas_characteristics = pd.DataFrame(dma_characts_json)
    dmas_characteristics.set_index('name_long', inplace=True)

    return dmas_characteristics

INPUT_DIR='input'
INFLOW_FILE='Inflows.xlsx'
WEATHER_FILE='Weather.xlsx'
WEATHER_FEATURES=['Rain', 'Temperature', 'Humidity', 'Windspeed']
WEATHER_UNITS=['mm', '°C', '%', 'km/h']

DATE='date'

def load_bwdf_final_data(data_dir:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    demands = pd.read_excel(os.path.join(data_dir, INPUT_DIR, INFLOW_FILE))
    weather = pd.read_excel(os.path.join(data_dir, INPUT_DIR, WEATHER_FILE))

    # basic preprocessing to set the first column as the index with name date
    def preprocess_date_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], format='%d/%m/%Y %H:%M')
        df = df.rename(columns={df.columns[0]: DATE})
        df = df.set_index(DATE)
        return df

    demands = preprocess_date_columns(demands)
    weather = preprocess_date_columns(weather)

    # Set the units of the columns for the weather data
    demands.columns = DMAS_NAMES
    weather.columns = WEATHER_FEATURES
    weather.attrs['units'] = dict(zip(WEATHER_FEATURES, WEATHER_UNITS))

    # Adjust the dataset to account for the missing hour in summer time and average the duplicates
    def adjust_summer_time(df: pd.DataFrame) -> pd.DataFrame:
        days_missing_hour = ['2021-03-28', '2022-03-27', '2023-03-26']

        # Copy 1AM and 3AM data to 2AM for days missing 2AM
        for day in days_missing_hour:
            df = pd.concat([df, df.loc[f'{day} 01:00:00':f'{day} 03:00:00']
                        .reset_index()
                        .assign(date=pd.to_datetime(f'{day} 02:00:00'))
                        .set_index(DATE)]) \
                            .sort_index()

        # Average 2AM values for days with duplicates
        return df.groupby(DATE).mean().sort_index()

    demands = adjust_summer_time(demands)
    weather = adjust_summer_time(weather)

    # Adjust the dataset to start on the first Monday
    def adjust_first_monday(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc['2021-01-04':]

    demands = adjust_first_monday(demands)
    weather = adjust_first_monday(weather)

    return demands, weather