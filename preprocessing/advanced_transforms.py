from preprocessing.base import Preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.stats as ss
import datetime
import statsmodels
from adtk.detector import SeasonalAD
import holidays
from Utils.weather_features import feels_like, wind_chill, heat_index, dew_point, Temp

DMAS_NAMES = ['DMA_A', 'DMA_B', 'DMA_C', 'DMA_D', 'DMA_E', 'DMA_F', 'DMA_G', 'DMA_H', 'DMA_I', 'DMA_J']

# AUX Functions
def load_characteristics():
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
    dmas_characteristics['mean_per_user'] = 100 * dmas_characteristics['h_mean'] / dmas_characteristics['population']
    dmas_characteristics.set_index('name_long', inplace=True)

    return dmas_characteristics

def decompose_into_n_signals(x, n):
    fourier = np.fft .rfft(x)
    frequencies = np.fft .rfftfreq(x.size, d=2e-2/x.size)
    out = []
    for vals in np.array_split(frequencies, n):
        ft_threshed = fourier.copy()
        ft_threshed[(vals.min() > frequencies)] = 0
        ft_threshed[(vals.max() < frequencies)] = 0
        out.append(np.fft .irfft(ft_threshed))
    return out

def get_lumpiness(x):
    v = [np.var(x_w) for x_w in np.array_split(x, len(x) // 24 + 1)]
    return np.var(v)

def get_stability(x):
    v = [np.mean(x_w) for x_w in np.array_split(x, len(x) // 24 + 1)]
    return np.var(v)

def dataset_week_number(a_date):
    """
    Return the number of the week in our dataset.
    The first 3 days are Friday, Saturday and Sunday so they are given the number 0.
    The fist week starting from Monday 4th is given the number 1 and so on.
    """
    begin_date = pd.to_datetime('2021-01-04')
    if a_date < begin_date:
        return 0
    return int((a_date - begin_date).days / 7) + 1
# AUX Functions


class LGBM_demand_features(Preprocessing):

    def transform(self, X):
        X['weekday'] = X.index.weekday + 1
        X['hour'] = X.index.hour
        X['holidays'] = 0
        it_holidays = holidays.country_holidays('IT')
        it_holidays = list(it_holidays.keys())
        it_holidays = set(it_holidays).intersection(X.index.date)
        X.loc[list(it_holidays), 'holidays'] = 1
        X.loc[X.index.date == datetime.date(2021, 11, 3), 'holidays'] = 1
        X.loc[X.index.date == datetime.date(2022, 11, 3), 'holidays'] = 1
        X['month'] = X.index.month
        X['season'] = pd.cut(
          (X.index.dayofyear + 11) % 366,
          [0, 91, 183, 275, 366],
          labels=[0, 1, 2, 3] # 'Winter', 'Spring', 'Summer', 'Fall'
        ).fillna(0)

        nan_cols = []
        for col in DMAS_NAMES:
            X['temp'] = 1 * X[col].isnull()
            X[col+'_nans'] = (X['temp'] * (X.groupby((X['temp'] != X['temp'].shift()).cumsum()).cumcount() + 1))
            nan_cols += [col+'_nans']

        # Standard encoding to capture hourly seasonality
        X['misc_hour_x'] = X['hour'].apply(lambda x:np.cos(2*np.pi*x/24))
        X['misc_hour_y'] = X['hour'].apply(lambda x:np.sin(2*np.pi*x/24))

        X['no_week'] = X.index.to_series().apply(lambda x:dataset_week_number(x))

        lagged_cols = []
        rest_lagged_cols = []

        dmas_characteristics = load_characteristics()

        for dma in DMAS_NAMES:

            X['temp'] = X.groupby(['hour'])[dma].shift(7)

            # Caclulate lumpiness: the variance of the chunk-wise variances. It captures potential changes in 2nd order effects.
            # Caclulate stability: the variance of the chunk-wise means. It captures potential changes in 1st order effects.
            X[dma+'_lumpiness_lagged'] = X[dma].rolling(24).apply(lambda x:get_lumpiness(x))
            X[dma+'_lumpiness_lagged'] = X.groupby(['hour'])[dma+'_lumpiness_lagged'].shift(7)
            X[dma+'_stability_lagged'] = X[dma].rolling(24).apply(lambda x:get_stability(x))
            X[dma+'_stability_lagged'] = X.groupby(['hour'])[dma+'_stability_lagged'].shift(7)
            lagged_cols += [dma+'_lumpiness_lagged',dma+'_stability_lagged']

            models = [OptimizedTheta(season_length=24, decomposition_type='multiplicative')]
            for w in list(np.sort(X['no_week'].unique())[2:-1]):
                # Exponential smoothing of the series. This feature together with the next 2, try to capture the standard patterns
                # of the series in different ways. It only uses data from the previous week. This is because the only seasonality within a week is a 24 hour
                # seasonality, whereas over longer horizons, things get less obvious
                model = ExponentialSmoothing(X.loc[X['no_week'] < w, dma].iloc[-WEEK_LEN:].reset_index(drop=True).astype('float64'), trend="mul", seasonal="mul", seasonal_periods=24)
                fit = model.fit()
                pred = fit.forecast(WEEK_LEN)
                X.loc[X['no_week'] == w, dma+'_smooth_lagged'] = pred.values

                # Use the theta model to decompose a series (in a similar logic to FFT) and forecast
                df_theta = X.loc[X['no_week'] < w, dma].iloc[-WEEK_LEN:].reset_index().rename(columns={'Date':'ds',dma:'y'})
                df_theta['unique_id'] = dma
                sf = StatsForecast(df=df_theta, models=models,freq='H',n_jobs=-1)
                sf.fit()
                pred = sf.forecast(WEEK_LEN, fitted=True)
                X.loc[X['no_week'] == w, dma+'_theta_lagged'] = pred['OptimizedTheta'].values

                # Something like a FFT, but it decomposes the series into two subseries: seasonal + trend and residuals. Those 3 features are
                # correlated but not too much (so they capture different aspects)
                dec = decompose_into_n_signals(X.loc[X['no_week'] < w, dma].iloc[-WEEK_LEN:].reset_index(drop=True).astype('float64'), 2)
                X.loc[X['no_week'] == w, dma+'_dec_0_lagged'] = dec[0]
                X.loc[X['no_week'] == w, dma+'_dec_1_lagged'] = dec[1]
            lagged_cols += [dma+'_dec_0_lagged', dma+'_smooth_lagged', dma+'_theta_lagged']

            if (dma == 'DMA_A'):
                # DMA A has a "strange" behavior where it has some patterns that occur again and again, and some that do not (noise).
                # dec_1 is the residual from FFT. Jarque bera identifies whether a series has a normal distribution. In the case of
                # DMA A is gives information of whether there were patters in the previous week
                X[dma+'_jb_lagged'] = X[dma+'_dec_1_lagged'].rolling(WEEK_LEN).apply(lambda x:ss.jarque_bera(x.values).statistic)
                lagged_cols += [dma+'_jb_lagged']

            # The ratio of current demand to the rolling 24 hour average. It captures regime changes
            lag = 24
            X[dma+'_/mean_'+str(lag)+'_lagged'] = X[dma] /  np.roll(np.append(np.convolve(X[dma], np.ones(lag)/lag, mode="valid"), np.ones(lag-1)), lag-1)
            X[dma+'_/mean_'+str(lag)+'_lagged'] = X.groupby(['hour'])[dma+'_/mean_'+str(lag)+'_lagged'].shift(7)
            lagged_cols += [dma+'_/mean_'+str(lag)+'_lagged']

            # The average, min and max over the last 4 weeks during the same hour and same day. It captures changes in seasonality.
            X[dma+'_lag4_avg_lagged'] = pd.concat([X.groupby(['hour'])[dma].shift(7), X.groupby(['hour'])[dma].shift(14), X.groupby(['hour'])[dma].shift(21), X.groupby(['hour'])[dma].shift(28)], axis=1).mean(axis=1)
            X[dma+'_lag4_min_lagged'] = pd.concat([X.groupby(['hour'])[dma].shift(7), X.groupby(['hour'])[dma].shift(14), X.groupby(['hour'])[dma].shift(21), X.groupby(['hour'])[dma].shift(28)], axis=1).min(axis=1)
            X[dma+'_lag4_max_lagged'] = pd.concat([X.groupby(['hour'])[dma].shift(7), X.groupby(['hour'])[dma].shift(14), X.groupby(['hour'])[dma].shift(21), X.groupby(['hour'])[dma].shift(28)], axis=1).max(axis=1)
            lagged_cols += [dma+'_lag4_avg_lagged',dma+'_lag4_min_lagged',dma+'_lag4_max_lagged']

            # The average, min and max over weeks 5-8 during the same hour and same day. It captures changes in seasonality.
            X[dma+'_lag8_avg_lagged'] = pd.concat([X.groupby(['hour'])[dma].shift(35), X.groupby(['hour'])[dma].shift(42), X.groupby(['hour'])[dma].shift(49), X.groupby(['hour'])[dma].shift(56)], axis=1).mean(axis=1)
            X[dma+'_lag8_min_lagged'] = pd.concat([X.groupby(['hour'])[dma].shift(35), X.groupby(['hour'])[dma].shift(42), X.groupby(['hour'])[dma].shift(49), X.groupby(['hour'])[dma].shift(56)], axis=1).min(axis=1)
            X[dma+'_lag8_max_lagged'] = pd.concat([X.groupby(['hour'])[dma].shift(35), X.groupby(['hour'])[dma].shift(42), X.groupby(['hour'])[dma].shift(49), X.groupby(['hour'])[dma].shift(56)], axis=1).max(axis=1)
            lagged_cols += [dma+'_lag8_avg_lagged',dma+'_lag8_min_lagged',dma+'_lag8_max_lagged']

            for i in [24,48,72,96,120,144]:
                X[dma+'_lag_'+str(i)] = X[dma].shift(WEEK_LEN+i)

            # Add some information about the demand during the same hour, for all days during the previous week.
            X[dma+'_lag_prev_week_avg_lagged'] = X[[dma+'_lag_'+str(i) for i in [24,48,72,96,120,144]]].mean(axis=1)
            lagged_cols += [dma+'_lag_prev_week_avg_lagged']


            # Some features that capture information for all dmas except for the chosen dma. All features below, try to bring 
            # information from other series in the form of average, min, max and std across the other DMAs. It is important to 
            # divide each DMA by the mean consumption per user in order for everything to be comparable. 
            rest_DMAs = list(set(DMAS_NAMES) - set([dma]))
            X['avg'] = X[rest_DMAs].div(dmas_characteristics.loc[dmas_characteristics.index.isin(rest_DMAs),'mean_per_user'].values, axis=1).mean(axis=1)

            X[dma+'_lag7_ewm_avg_rest'] = X.groupby(['hour'])['avg'].shift(7).ewm(alpha=0.6).mean()
            X[dma+'_lag7_ewm_std_rest'] = X.groupby(['hour'])['avg'].shift(7).ewm(alpha=0.6).std()
            rest_lagged_cols += [dma+'_lag7_ewm_avg_rest',dma+'_lag7_ewm_std_rest']

            X['min'] = X[rest_DMAs].div(dmas_characteristics.loc[dmas_characteristics.index.isin(rest_DMAs),'mean_per_user'].values, axis=1).min(axis=1)
            X['max'] = X[rest_DMAs].div(dmas_characteristics.loc[dmas_characteristics.index.isin(rest_DMAs),'mean_per_user'].values, axis=1).max(axis=1)
            X[dma+'_lag4_min_avg_rest'] = pd.concat([X.groupby(['hour'])['min'].shift(7), X.groupby(['hour'])['min'].shift(14), X.groupby(['hour'])['min'].shift(21), X.groupby(['hour'])['min'].shift(28)], axis=1).mean(axis=1)
            X[dma+'_lag4_max_avg_rest'] = pd.concat([X.groupby(['hour'])['max'].shift(7), X.groupby(['hour'])['max'].shift(14), X.groupby(['hour'])['max'].shift(21), X.groupby(['hour'])['max'].shift(28)], axis=1).mean(axis=1)
            rest_lagged_cols += [dma+'_lag4_min_avg_rest',dma+'_lag4_max_avg_rest']

        misc_cols = ['misc_hour_x','misc_hour_y']
        categorical_feats = ['hour','weekday','holidays','season']
        X[categorical_feats] = X[categorical_feats].astype('category')

        return X[DMAS_NAMES + misc_cols + categorical_feats + lagged_cols + rest_lagged_cols + nan_cols]


class LGBM_weather_features(Preprocessing):

    def transform(self, X):
        X['temp'] = X['Rain'].eq(0)
        X['days_since_rain'] = (X['temp'] * (X.groupby((X['temp'] != X['temp'].shift()).cumsum()).cumcount() + 1))
        X.drop(columns=['temp'], inplace=True)

        # Some weather features that combine information from the given weather features in a more intuitive manner
        X['real_feel'] = X[['Temperature','Humidity','Windspeed']].apply(lambda x:feels_like(temperature=Temp(x[0], 'c'), humidity=x[1], wind_speed=x[2]).c, axis=1)
        X['wind_chill'] = X[['Temperature','Humidity','Windspeed']].apply(lambda x:wind_chill(temperature=Temp(x[0], 'c'), wind_speed=x[2]).c, axis=1)
        X['heat_index'] = X[['Temperature','Humidity','Windspeed']].apply(lambda x:heat_index(temperature=Temp(x[0], 'c'), humidity=x[1]).c, axis=1)
        X['dew_point'] = X[['Temperature','Humidity','Windspeed']].apply(lambda x:dew_point(temperature=Temp(x[0], 'c'), humidity=x[1]).c, axis=1)
        X['cdd'] = X.loc[X['Temperature'] < 18, 'Temperature'].transform(lambda x:(18 - x).rolling(24).mean())
        X['cdd'] = X['cdd'].fillna(0)

        weather_cols = ['Rain','real_feel','wind_chill','heat_index','dew_point','cdd','days_since_rain']

        X['hour'] = X.index.hour
        weather_lagged_cols = []
        # For the following weather features, get the average values across the same hour over the last 4 days (captures changes in seasonality)
        # and the average during the last four days (to capture changes in mean)
        w_cols = ['real_feel','heat_index','wind_chill','dew_point']
        for col in w_cols:
            X[col+'_lag4_avg'] = X.groupby(['hour'])[col].shift(1).rolling(4).mean()
            X[col+'_lag4_avg24'] = X[col].rolling(24*4).mean()
            weather_lagged_cols += [col+'_lag4_avg',col+'_lag4_avg24']

        return X[weather_cols + weather_lagged_cols]


class LGBM_impute_nan_demand(Preprocessing):

    def transform(self, X):

        detector = SeasonalAD(c=24)

        for dma in DMAS_NAMES:
            # Detect anomalies
            anomalies = detector.fit_predict(X[dma].ffill())
            X.loc[anomalies.fillna(False).values, dma] = np.nan

        X = X.apply(lambda x:x.interpolate(limit=1))
        X['hour'] = X.index.hour
        X[X.columns[:-1]] = X.groupby(["hour"]).transform(lambda x: x.fillna(x.rolling(4).mean()))
        X[X.columns[:-1]] = X.groupby(["hour"]).transform(lambda x: x.fillna(x.mean()))

        return X.drop(columns=['hour'])


class LGBM_impute_nan_weather(Preprocessing):

    def transform(self, X):

        X = X.apply(lambda x:x.interpolate(limit=1))
        X['hour'] = X.index.hour
        X[X.columns[:-1]] = X.groupby(["hour"]).transform(lambda x: x.fillna(x.rolling(4).mean()))
        X[X.columns[:-1]] = X.groupby(["hour"]).transform(lambda x: x.fillna(x.mean()))

        return X.drop(columns=['hour'])


class LGBM_prepare_test_dfs(Preprocessing):

    def transform(self, demand_train, weather_train, weather_test):
        demand_nans = pd.DataFrame(columns=demand_train.columns, data=np.nan, index=weather_test.index)
        weather_nans = pd.DataFrame(columns=weather_train.columns, data=np.nan, index=weather_test.index)
        weather_nans.loc[:, weather_test.columns] = weather_test.loc[:, weather_test.columns]

        demands_test = pd.concat([demand_train, demand_nans], axis=0)
        weather_test = pd.concat([weather_train, weather_nans], axis=0)

        return demands_test, weather_test
