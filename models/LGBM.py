from models.base import Model
import numpy as np
from preprocessing.simple_transforms import Logarithm
from preprocessing.advanced_transforms import  LGBM_demand_features, LGBM_weather_features, LGBM_impute_nan_demand, LGBM_impute_nan_weather, LGBM_prepare_test_dfs
import lightgbm as lgb
import xgboost as xgb
import random
import warnings
warnings.filterwarnings("ignore")

DMAS_NAMES = ['DMA_A', 'DMA_B', 'DMA_C', 'DMA_D', 'DMA_E', 'DMA_F', 'DMA_G', 'DMA_H', 'DMA_I', 'DMA_J']
WEEK_LEN = 24 * 7
RANDOM_SEED = 46
random.seed(RANDOM_SEED)


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

# No hyperparameter tuning for all parameters
lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'num_leaves': 32,
        'max_depth': 6,
        'learning_rate': 0.01,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.8,
        'bagging_freq':10,
        'verbose': -1
}
xgb_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.02,
    'max_depth': 6,
    'subsample': 0.8,
    'objective':'reg:squarederror',
    'min_child_weight':10,
    'silent':1
}

lgbm_simple = {
    'name': 'LGBMsimple',
    'model': LGBMsimple(lgb_params = lgb_params),
    'preprocessing': {
        'demand': [Logarithm(), LGBM_impute_nan_demand(), LGBM_demand_features(no_last_week=1)],
        'weather': [LGBM_impute_nan_weather(), LGBM_weather_features()],
        'prepare_test_dfs': [LGBM_prepare_test_dfs()]
    }
}
xgbm_simple = {
    'name': 'XGBMsimple',
    'model': XGBMsimple(xgb_params = xgb_params),
    'preprocessing': {
        'demand': [Logarithm(), LGBM_impute_nan_demand(), LGBM_demand_features(no_last_week=0)],
        'weather': [LGBM_impute_nan_weather(), LGBM_weather_features()],
        'prepare_test_dfs': [LGBM_prepare_test_dfs()]
    }
}
lgbm_robust = {
    'name': 'LGBMrobust',
    'model': LGBMrobust(lgb_params = lgb_params),
    'preprocessing': {
        'demand': [Logarithm(), LGBM_impute_nan_demand(), LGBM_demand_features(no_last_week=1)],
        'weather': [LGBM_impute_nan_weather(), LGBM_weather_features()],
        'prepare_test_dfs': [LGBM_prepare_test_dfs()]
    }
}
lgbm_simple_with_last_week = {
    'name': 'LGBMsimple_with_last week',
    'model': LGBMsimple(lgb_params = lgb_params),
    'preprocessing': {
        'demand': [Logarithm(), LGBM_impute_nan_demand(), LGBM_demand_features(no_last_week=0)],
        'weather': [LGBM_impute_nan_weather(), LGBM_weather_features()],
        'prepare_test_dfs': [LGBM_prepare_test_dfs()]
    }
}
