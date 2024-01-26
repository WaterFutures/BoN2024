from models.base import Model
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

DMAS_NAMES = ['DMA_A', 'DMA_B', 'DMA_C', 'DMA_D', 'DMA_E', 'DMA_F', 'DMA_G', 'DMA_H', 'DMA_I', 'DMA_J']
WEEK_LEN = 24 * 7

class LGBMrobust(Model):
    def __init__(self, lgb_params) -> None:
        super().__init__()
        self.model = None
        self.lgb_params = lgb_params
        self.feats = {}
        self.model = {}

    def fit(self, demands, weather):
        self.dmas = DMAS_NAMES

        X_train = demands.merge(weather, right_index=True, left_index=True)

        nan_cols = [c for c in X_train.columns if c.endswith('_nans')]
        misc_cols = [c for c in X_train.columns if c.startswith('misc_')]
        rest_lagged_cols = [c for c in X_train.columns if c.endswith('_rest')]
        lagged_cols = [c for c in X_train.columns if c.endswith('_lagged')]
        weather_cols = weather.columns.tolist()
        cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'category']

        for dma in self.dmas:
            self.feats[dma] = [c for c in lagged_cols + nan_cols if c.startswith(dma)] + weather_cols + cat_cols + misc_cols + [c for c in rest_lagged_cols if c.startswith(dma)]
            for m in range(0,10):
                train_set = lgb.Dataset(X_train[self.feats[dma]], label=X_train[dma] + np.random.normal(0, X_train[dma].std() * 0.05, X_train.shape[0]), categorical_feature=cat_cols, free_raw_data=False)
                self.model[dma+'_'+str(m)] = lgb.train(self.lgb_params, train_set, num_boost_round=1000, categorical_feature=cat_cols)

    def forecast(self, demand_test, weather_test):
        demand_test = demand_test.iloc[-WEEK_LEN:,:]
        weather_test = weather_test.iloc[-WEEK_LEN:,:]
        X_test = demand_test.merge(weather_test, right_index=True, left_index=True)
        pred = np.array([np.mean(np.array([self.model[dma+'_'+str(m)].predict(X_test[self.feats[dma]]) for m in range(0,10)]), axis=0) for dma in self.dmas])
        return pred.T


class LGBMsimple(Model):
    def __init__(self, lgb_params) -> None:
        super().__init__()
        self.model = None
        self.lgb_params = lgb_params
        self.feats = {}
        self.model = {}

    def fit(self, demands, weather):
        self.dmas = DMAS_NAMES

        X_train = demands.merge(weather, right_index=True, left_index=True)

        nan_cols = [c for c in X_train.columns if c.endswith('_nans')]
        misc_cols = [c for c in X_train.columns if c.startswith('misc_')]
        rest_lagged_cols = [c for c in X_train.columns if c.endswith('_rest')]
        lagged_cols = [c for c in X_train.columns if c.endswith('_lagged')]
        weather_cols = weather.columns.tolist()
        cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'category']

        for dma in self.dmas:
            self.feats[dma] = [c for c in lagged_cols + nan_cols if c.startswith(dma)] + weather_cols + cat_cols + misc_cols + [c for c in rest_lagged_cols if c.startswith(dma)]

            train_set = lgb.Dataset(X_train[self.feats[dma]], label=X_train[dma], categorical_feature=cat_cols, free_raw_data=False)
            self.model[dma] = lgb.train(self.lgb_params, train_set, num_boost_round=1000, categorical_feature=cat_cols)

    def forecast(self, demand_test, weather_test):
        demand_test = demand_test.iloc[-WEEK_LEN:,:]
        weather_test = weather_test.iloc[-WEEK_LEN:,:]
        X_test = demand_test.merge(weather_test, right_index=True, left_index=True)
        pred = np.array([self.model[dma].predict(X_test[self.feats[dma]]) for dma in self.dmas])
        return pred.T

# Added also the XGB model (very similar to LGB but it can be used in the ensembling step)
class XGBMsimple(Model):
    def __init__(self, xgb_params) -> None:
        super().__init__()
        self.model = None
        self.xgb_params = xgb_params
        self.feats = {}
        self.model = {}

    def fit(self, demands, weather):
        self.dmas = DMAS_NAMES

        X_train = demands.merge(weather, right_index=True, left_index=True)

        nan_cols = [c for c in X_train.columns if c.endswith('_nans')]
        misc_cols = [c for c in X_train.columns if c.startswith('misc_')]
        rest_lagged_cols = [c for c in X_train.columns if c.endswith('_rest')]
        lagged_cols = [c for c in X_train.columns if c.endswith('_lagged')]
        weather_cols = weather.columns.tolist()
        cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'category']

        for dma in self.dmas:
            self.feats[dma] = [c for c in lagged_cols + nan_cols if c.startswith(dma)] + weather_cols + cat_cols + misc_cols + [c for c in rest_lagged_cols if c.startswith(dma)]

            train_set = xgb.DMatrix(X_train[self.feats[dma]], X_train[dma], enable_categorical=True)
            self.model[dma] = xgb.train(self.xgb_params, train_set, num_boost_round=1000)

    def forecast(self, demand_test, weather_test):
        demand_test = demand_test.iloc[-WEEK_LEN:,:]
        weather_test = weather_test.iloc[-WEEK_LEN:,:]
        X_test = demand_test.merge(weather_test, right_index=True, left_index=True)
        pred = np.array([self.model[dma].predict(xgb.DMatrix(X_test[self.feats[dma]], enable_categorical=True)) for dma in self.dmas])
        return pred.T

