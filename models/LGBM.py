from models.base import Model
import numpy as np
from preprocessing.simple_transforms import Logarithm
from preprocessing.advanced_transforms import  LGBM_demand_features, LGBM_weather_features, LGBM_impute_nan_demand, LGBM_impute_nan_weather, LGBM_prepare_test_dfs
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

DMAS_NAMES = ['DMA_A', 'DMA_B', 'DMA_C', 'DMA_D', 'DMA_E', 'DMA_F', 'DMA_G', 'DMA_H', 'DMA_I', 'DMA_J']


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
        X_test = demand_test.merge(weather_test, right_index=True, left_index=True)
        pred = np.array([self.model[dma].predict(X_test[self.feats[dma]]) for dma in self.dmas])
        return pred.T


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
lgbm_simple = {
    'name': 'LGBMsimple',
    'model': LGBMsimple(lgb_params = lgb_params),
    'preprocessing': {
        'demand': [Logarithm(), LGBM_impute_nan_demand(), LGBM_demand_features()],
        'weather': [LGBM_impute_nan_weather(), LGBM_weather_features()],
        'prepare_test_dfs': [LGBM_prepare_test_dfs()]
    }
}
