from model import Model
from data_loader import DMAS_NAMES, load_characteristics
from constants import WEEK_LEN
import pandas as pd
import numpy as np
from Utils.weather_features import feels_like, wind_chill, heat_index, dew_point, Temp

import scipy.stats as ss
import statsmodels
from adtk.detector import SeasonalAD
import holidays
import datetime
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


def prepare_df_(df, test=False):

    df['week'] = df.index.isocalendar().week
    df['weekday'] = df.index.weekday + 1
    df['year'] = df.index.year
    df['hour'] = df.index.hour

    df['holidays'] = 0
    it_holidays = holidays.country_holidays('IT')
    it_holidays = list(it_holidays.keys())
    it_holidays = set(it_holidays).intersection(df.index.date)
    df.loc[list(it_holidays), 'holidays'] = 1
    df.loc[df.index.date == datetime.date(2021, 11, 3), 'holidays'] = 1
    df.loc[df.index.date == datetime.date(2022, 11, 3), 'holidays'] = 1
    df['month'] = df.index.month
    df['season'] = pd.cut(
        (df.index.dayofyear + 11) % 366,
        [0, 91, 183, 275, 366],
        labels=[0, 1, 2, 3] # 'Winter', 'Spring', 'Summer', 'Fall'
    ).fillna(0)
    df['weekend'] = 0
    df.loc[df['weekday'] > 5,'weekend'] = 1

    df['is_month_start'] = 1 * df.index.is_month_start
    df['is_month_end'] = 1 * df.index.is_month_end

    if test:
        for col in DMAS_NAMES:
            df[col+'_nans'] = 0
    else:
        for col in DMAS_NAMES:
            df['temp'] = 1 * df[col].isnull()
            df[col+'_nans'] = (df['temp'] * (df.groupby((df['temp'] != df['temp'].shift()).cumsum()).cumcount() + 1))
            df.drop(columns=['temp'], inplace=True)

    return df

def impute_nan_(df):

    detector = SeasonalAD(c=24)

    for col in DMAS_NAMES:
        # Detect anomalies
        anomalies = detector.fit_predict(df[col].ffill())
        df.loc[anomalies.fillna(False).values, col] = np.nan

    weather_cols = ['Rain', 'Temperature', 'Humidity', 'Windspeed']
    df[DMAS_NAMES + weather_cols] = df[DMAS_NAMES + weather_cols].apply(lambda x:x.interpolate(limit=1))
    df[DMAS_NAMES + weather_cols] = df.groupby(["hour"])[DMAS_NAMES + weather_cols].transform(lambda x: x.fillna(x.rolling(4).mean()))
    df[DMAS_NAMES + weather_cols] = df.groupby(["hour"])[DMAS_NAMES + weather_cols].transform(lambda x: x.fillna(x.mean()))

    return df


def decompose_into_n_signals(srs, n):
    fourier = np.fft .rfft(srs)
    frequencies = np.fft .rfftfreq(srs.size, d=2e-2/srs.size)
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

def add_customized_feats(df_actual):

    df_actual['temp'] = df_actual['Rain'].eq(0)
    df_actual['days_since_rain'] = (df_actual['temp'] * (df_actual.groupby((df_actual['temp'] != df_actual['temp'].shift()).cumsum()).cumcount() + 1))
    df_actual.drop(columns=['temp'], inplace=True)

    df_actual['hour_x'] = df_actual['hour'].apply(lambda x:np.cos(2*np.pi*x/24))
    df_actual['hour_y'] = df_actual['hour'].apply(lambda x:np.sin(2*np.pi*x/24))
    other_cols = ['hour_x','hour_y','days_since_rain']

    df_actual['real_feel'] = df_actual[['Temperature','Humidity','Windspeed']].apply(lambda x:feels_like(temperature=Temp(x[0], 'c'), humidity=x[1], wind_speed=x[2]).c, axis=1)
    df_actual['wind_chill'] = df_actual[['Temperature','Humidity','Windspeed']].apply(lambda x:wind_chill(temperature=Temp(x[0], 'c'), wind_speed=x[2]).c, axis=1)
    df_actual['heat_index'] = df_actual[['Temperature','Humidity','Windspeed']].apply(lambda x:heat_index(temperature=Temp(x[0], 'c'), humidity=x[1]).c, axis=1)
    df_actual['dew_point'] = df_actual[['Temperature','Humidity','Windspeed']].apply(lambda x:dew_point(temperature=Temp(x[0], 'c'), humidity=x[1]).c, axis=1)
    df_actual['cdd'] = df_actual.loc[df_actual['Temperature'] < 18, 'Temperature'].transform(lambda x:(18 - x).rolling(24).mean())
    df_actual['cdd'] = df_actual['cdd'].fillna(0)

    new_weather_cols = ['Rain','real_feel','wind_chill','heat_index','dew_point','cdd']

    dmas_characteristics = load_characteristics()

    lagged_cols = []
    rest_lagged_cols = []

    df_actual[DMAS_NAMES] = np.log1p(df_actual[DMAS_NAMES])

    for dma in DMAS_NAMES:

        df_actual['temp'] = df_actual.groupby(['hour'])[dma].shift(7)

        df_actual[dma+'_lumpiness'] = df_actual[dma].rolling(24).apply(lambda x:get_lumpiness(x))
        df_actual[dma+'_lumpiness'] = df_actual.groupby(['hour'])[dma+'_lumpiness'].shift(7)
        df_actual[dma+'_stability'] = df_actual[dma].rolling(24).apply(lambda x:get_stability(x))
        df_actual[dma+'_stability'] = df_actual.groupby(['hour'])[dma+'_stability'].shift(7)
        lagged_cols += [dma+'_lumpiness',dma+'_stability']

        for w in list(np.sort(df_actual['no_week'].unique())[4:-1]):
            model = statsmodels.tsa.holtwinters.ExponentialSmoothing(df_actual.loc[df_actual['no_week'] < w, dma].iloc[-WEEK_LEN:].reset_index(drop=True).astype('float64'), trend="mul", seasonal="mul", seasonal_periods=24)
            fit = model.fit()
            pred = fit.forecast(WEEK_LEN)
            df_actual.loc[df_actual['no_week'] == w, dma+'_smooth'] = pred.values
        lagged_cols += [dma+'_smooth']


        for w in list(np.sort(df_actual['no_week'].unique())):
            dec = decompose_into_n_signals(df_actual.loc[df_actual['no_week'] == w, 'temp'], 2)
            df_actual.loc[df_actual['no_week'] == w, dma+'_dec_0'] = dec[0]
            df_actual.loc[df_actual['no_week'] == w, dma+'_dec_1'] = dec[1]
        df_actual.drop(columns=['temp'], inplace=True)

        lagged_cols += [dma+'_dec_0']

        if (dma == 'DMA_A'):
            df_actual[dma+'_jb'] = df_actual[dma+'_dec_1'].rolling(WEEK_LEN).apply(lambda x:ss.jarque_bera(x.values).statistic)
            lagged_cols += [dma+'_jb']

        lag = 24
        df_actual[dma+'_/mean_'+str(lag)] = df_actual[dma] /  np.roll(np.append(np.convolve(df_actual[dma], np.ones(lag)/lag, mode="valid"), np.ones(lag-1)), lag-1)
        df_actual[dma+'_/mean_'+str(lag)] = df_actual.groupby(['hour'])[dma+'_/mean_'+str(lag)].shift(7)
        lagged_cols += [dma+'_/mean_'+str(lag)]


        df_actual[dma+'_lag7_avg'] = pd.concat([df_actual.groupby(['hour'])[dma].shift(7), df_actual.groupby(['hour'])[dma].shift(14), df_actual.groupby(['hour'])[dma].shift(21), df_actual.groupby(['hour'])[dma].shift(28)], axis=1).mean(axis=1)
        df_actual[dma+'_lag7_min'] = pd.concat([df_actual.groupby(['hour'])[dma].shift(7), df_actual.groupby(['hour'])[dma].shift(14), df_actual.groupby(['hour'])[dma].shift(21), df_actual.groupby(['hour'])[dma].shift(28)], axis=1).min(axis=1)
        df_actual[dma+'_lag7_max'] = pd.concat([df_actual.groupby(['hour'])[dma].shift(7), df_actual.groupby(['hour'])[dma].shift(14), df_actual.groupby(['hour'])[dma].shift(21), df_actual.groupby(['hour'])[dma].shift(28)], axis=1).max(axis=1)
        lagged_cols += [dma+'_lag7_avg',dma+'_lag7_min',dma+'_lag7_max']


        for i in [24,48,72,96,120,144]:
            df_actual[dma+'_lag_daily_'+str(i)] = df_actual[dma].shift(WEEK_LEN+i)

        df_actual[dma+'_lag_daily_avg'] = df_actual[[dma+'_lag_daily_'+str(i) for i in [24,48,72,96,120,144]]].mean(axis=1)
        lagged_cols += [dma+'_lag_daily_avg']


        rest_DMAs = np.sort(list(set(DMAS_NAMES) - set([dma])))
        df_actual['avg'] = df_actual[rest_DMAs].div(dmas_characteristics.loc[dmas_characteristics.index.isin(rest_DMAs),'mean_per_user'].values, axis=1).mean(axis=1)

        df_actual[dma+'_rest_lag7_ewm_avg'] = df_actual.groupby(['hour'])['avg'].shift(7).ewm(alpha=0.6).mean()
        df_actual[dma+'_rest_lag7_ewm_std'] = df_actual.groupby(['hour'])['avg'].shift(7).ewm(alpha=0.6).std()
        rest_lagged_cols += [dma+'_rest_lag7_ewm_avg',dma+'_rest_lag7_ewm_std']

        df_actual['min'] = df_actual[rest_DMAs].div(dmas_characteristics.loc[dmas_characteristics.index.isin(rest_DMAs),'mean_per_user'].values, axis=1).min(axis=1)
        df_actual['max'] = df_actual[rest_DMAs].div(dmas_characteristics.loc[dmas_characteristics.index.isin(rest_DMAs),'mean_per_user'].values, axis=1).max(axis=1)
        df_actual[dma+'_rest_lag7_min_avg'] = pd.concat([df_actual.groupby(['hour'])['min'].shift(7), df_actual.groupby(['hour'])['min'].shift(14), df_actual.groupby(['hour'])['min'].shift(21), df_actual.groupby(['hour'])['min'].shift(28)], axis=1).mean(axis=1)
        df_actual[dma+'_rest_lag7_max_avg'] = pd.concat([df_actual.groupby(['hour'])['max'].shift(7), df_actual.groupby(['hour'])['max'].shift(14), df_actual.groupby(['hour'])['max'].shift(21), df_actual.groupby(['hour'])['max'].shift(28)], axis=1).mean(axis=1)
        rest_lagged_cols += [dma+'_rest_lag7_min_avg',dma+'_rest_lag7_max_avg']


    weather_lagged_cols = []
    w_cols = ['heat_index','wind_chill','dew_point']

    for col in w_cols:
        df_actual[col+'_lag7_avg'] = df_actual.groupby(['hour'])[col].shift(1).rolling(4).mean()
        df_actual[col+'_lag7_avg24'] = df_actual[col].rolling(24*4).mean()

        weather_lagged_cols += [col+'_lag7_avg',col+'_lag7_avg24']

    new_categorical_feats = ['hour','weekday','holidays','season']

    return df_actual, lagged_cols, rest_lagged_cols, weather_lagged_cols, new_weather_cols, other_cols, new_categorical_feats



class LGBMsimple(Model):
    def __init__(self, lgb_params) -> None:
        super().__init__()
        self.model = None
        self.lgb_params = lgb_params
        self.feat_cat_cols = {}
        self.feats_1_2 = {}
        self.model_1_2 = {}
        self.feats_3 = {}
        self.model_3 = {}
        self.y_train = {}

    def name (self) -> str:
        return "LGBM"

    def forecasted_dmas(self) -> list:
        return DMAS_NAMES

    def preprocess_data(self,
                        train__dmas_h_q: pd.DataFrame,
                        train__exin_h:pd.DataFrame,
                        test__exin_h:pd.DataFrame) -> tuple [pd.DataFrame, pd.DataFrame]:

        df_train = train__dmas_h_q.merge(train__exin_h.drop(columns='no_week'), right_index=True, left_index=True)
        df_train = prepare_df_(df_train)
        df_train_c = df_train.copy()
        df_train_c = impute_nan_(df_train_c)

        df_train_c, lagged_cols, rest_lagged_cols, weather_lagged_cols, new_weather_cols, other_cols, new_categorical_feats = add_customized_feats(df_train_c)
        df_train_c[new_categorical_feats] = df_train_c[new_categorical_feats].astype('category')
        df_nan_test = pd.DataFrame(columns=df_train.columns, data=np.nan, index=test__exin_h.index)
        df_nan_test.loc[:,['no_week', 'Rain', 'Temperature', 'Humidity', 'Windspeed']] = test__exin_h.loc[:, ['no_week', 'Rain', 'Temperature', 'Humidity', 'Windspeed']]
        df_nan_test = prepare_df_(df_nan_test, test=True)
        df_test_c = pd.concat([df_train, df_nan_test],axis=0).copy()

        df_test_c = impute_nan_(df_test_c)
        df_test_c, _, _, _, _, _, _ = add_customized_feats(df_test_c)
        df_test_c[new_categorical_feats] = df_test_c[new_categorical_feats].astype('category')

        nan_cols = [c for c in df_train_c.columns if c.endswith('_nans')]
        feats = lagged_cols + rest_lagged_cols + weather_lagged_cols + new_weather_cols + new_categorical_feats + nan_cols+ other_cols

        for dma in DMAS_NAMES:
            self.feats_1_2[dma] = [c for c in lagged_cols if c.startswith(dma)] + weather_lagged_cols + new_weather_cols + new_categorical_feats + [c for c in nan_cols if c.startswith(dma)] + other_cols
            self.feats_3[dma] = self.feats_1_2[dma] + [c for c in rest_lagged_cols if c.startswith(dma)]
            self.feat_cat_cols[dma] = new_categorical_feats

        return (df_train_c, df_test_c.iloc[-WEEK_LEN:,:])

    def fit(self, X_train: pd.DataFrame) -> None:
        for dma in DMAS_NAMES:
            train_set = lgb.Dataset(X_train[self.feats_1_2[dma]], label=X_train[dma], categorical_feature=self.feat_cat_cols[dma], free_raw_data=False)
            self.model_1_2[dma] = lgb.train(self.lgb_params, train_set, num_boost_round=1000, categorical_feature=self.feat_cat_cols[dma])
            train_set = lgb.Dataset(X_train[self.feats_3[dma]], label=X_train[dma], categorical_feature=self.feat_cat_cols[dma], free_raw_data=False)
            self.model_3[dma] = lgb.train(self.lgb_params, train_set, num_boost_round=1000, categorical_feature=self.feat_cat_cols[dma])

    def forecast(self, X_test: pd.DataFrame) -> np.ndarray:
        for dma in DMAS_NAMES:
            pred = np.array([np.concatenate((np.expm1(self.model_1_2[dma].predict(X_test[self.feats_1_2[dma]].iloc[-24*7:-24*6,:])), np.expm1(self.model_3[dma].predict(X_test[self.feats_3[dma]].iloc[-24*6:,:]))), axis=0) for dma in DMAS_NAMES])

        return pred.T
