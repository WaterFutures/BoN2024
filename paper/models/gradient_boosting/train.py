from wflib.evaluator import WaterFuturesEvaluator as WFE

from lightgmb_impl import LGBMsimple, LGBMrobust
from xgboost_impl import XGBMsimple
from preprocessing.simple_transforms import Logarithm
from preprocessing.advanced_transforms import LGBM_demand_features, LGBM_impute_nan_demand
from preprocessing.advanced_transforms import LGBM_impute_nan_weather, LGBM_weather_features

wfe = WFE()

################################################################################
# LightGBM
################################################################################
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

wfe.add_model_configuration({
    'name': 'lgbm_simple',
    'model': LGBMsimple(lgb_params = lgb_params),
    'preprocessing': {
        'target': [Logarithm(), LGBM_impute_nan_demand(), LGBM_demand_features(no_last_week=1)],
        'exogenous': [LGBM_impute_nan_weather(), LGBM_weather_features()],
        'prediction_requires_extended_dfs': True
    },
    'deterministic': False,
    'n_eval_runs': 5
})
wfe.add_model_configuration({
    'name': 'lgmb_robust',
    'model': LGBMrobust(lgb_params = lgb_params),
    'preprocessing': {
        'target': [Logarithm(), LGBM_impute_nan_demand(), LGBM_demand_features(no_last_week=1)],
        'exogenous': [LGBM_impute_nan_weather(), LGBM_weather_features()],
        'prediction_requires_extended_dfs': True
    },
    'deterministic': False,
    'n_eval_runs': 5
})
wfe.add_model_configuration({
    'name': 'lgmb_simple_with_last_week',
    'model': LGBMsimple(lgb_params = lgb_params),
    'preprocessing': {
        'target': [Logarithm(), LGBM_impute_nan_demand(), LGBM_demand_features(no_last_week=0)],
        'exogenous': [LGBM_impute_nan_weather(), LGBM_weather_features()],
        'prediction_requires_extended_dfs': True
    },
    'deterministic': False,
    'n_eval_runs': 5
})

################################################################################
# XGBOOST
################################################################################
xgb_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.02,
    'max_depth': 6,
    'subsample': 0.8,
    'objective':'reg:squarederror',
    'min_child_weight':10,
    'silent':1
}

wfe.add_model_configuration({
    'name': 'xgbm_simple',
    'model': XGBMsimple(xgb_params = xgb_params),
    'preprocessing': {
        'target': [Logarithm(), LGBM_impute_nan_demand(), LGBM_demand_features(no_last_week=0)],
        'exogenous': [LGBM_impute_nan_weather(), LGBM_weather_features()],
        'prediction_requires_extended_dfs': True
    },
    'deterministic': False,
    'n_eval_runs': 5
})