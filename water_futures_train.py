# prepare the workspace
import eval
from eval.evaluator import WaterFuturesEvaluator
from eval.dashboard import run_dashboard

# prepare the evaluator
wfe = WaterFuturesEvaluator()

# Prepare the evaluator for the next iteration
wfe.next_iter()
# Collect all the models and the settings that we are considering
import models
import preprocessing

# Prepare the models
from models.benchmark import RollingAverageWeek, AutoRollingAverageWeek
from preprocessing.impute_and_fill import FillZero, FillAvgWeek

previous_week = {
    'name': 'PrevWeek',
    'model': RollingAverageWeek(1),
    'preprocessing': {
        'demand': [FillZero()],
        'weather': []
    },
    'deterministic': True
}
previous_week_v2 = {
    'name': 'PrevWeek_v2',
    'model': RollingAverageWeek(1),
    'preprocessing': {
        'demand': [FillAvgWeek()],
        'weather': []
    },
    'deterministic': True
}

average_week = {
    'name': 'AvgWeek',
    'model': RollingAverageWeek(None),
    'preprocessing': {
        'demand': [],
        'weather': []
    },
    'deterministic': True
}

rolling_average_2 = {
    'name': 'RollingAverage_2',
    'model': RollingAverageWeek(2),
    'preprocessing': {
        'demand': [FillZero()],
        'weather': []
    },
    'deterministic': True
}

rolling_average_4 = {
    'name': 'RollingAverage_4',
    'model': RollingAverageWeek(4),
    'preprocessing': {
        'demand': [FillZero()],
        'weather': []
    },
    'deterministic': True
}

rolling_average_8 = {
    'name': 'RollingAverage_8',
    'model': RollingAverageWeek(8),
    'preprocessing': {
        'demand': [FillZero()],
        'weather': []
    },
    'deterministic': True
}

auto_rollaw = {
    'name': 'AutoRollingAverage',
    'model': AutoRollingAverageWeek(),
    'preprocessing': {
        'demand': [FillAvgWeek()],
        'weather': []
    },
    'deterministic': False
}

models_configs = [
    previous_week,
    previous_week_v2,
    average_week,
    rolling_average_2,
    rolling_average_4,
    rolling_average_8,
    auto_rollaw
]

from models.exp_rolling_average_week import ExpWeightedRollingWeek

exp_rolling_average_2 = {
    'name': 'ExpRollingAverage_2',
    'model': ExpWeightedRollingWeek(2),
    'preprocessing': {
        'demand': [FillAvgWeek()],
        'weather': []
    },
    'deterministic': True
}

exp_rolling_average_4 = {
    'name': 'ExpRollingAverage_4',
    'model': ExpWeightedRollingWeek(4),
    'preprocessing': {
        'demand': [FillAvgWeek()],
        'weather': []
    },
    'deterministic': True
}

exp_rolling_average_8 = {
    'name': 'ExpRollingAverage_8',
    'model': ExpWeightedRollingWeek(8),
    'preprocessing': {
        'demand': [FillAvgWeek()],
        'weather': []
    },
    'deterministic': True
}

models_configs += [
    exp_rolling_average_2,
    exp_rolling_average_4,
    exp_rolling_average_8
]

from models.pattern_regression import PatternRegression, PatternRegressionDaily
from preprocessing.simple_transforms import Logarithm
from preprocessing.weather_feature_engineering import RealFeel, DewPoint, WindChill

pattern_regression = {
    'name': f'PatternRegression',
    'model': PatternRegression(),
    'preprocessing': {
        'demand': [Logarithm()],
        'weather': [RealFeel(), DewPoint(), WindChill()]
    },
    'deterministic': True
}

pattern_regression_daily = {
    'name': f'PatternRegressionDaily',
    'model': PatternRegressionDaily(),
    'preprocessing': {
        'demand': [Logarithm()],
        'weather': [RealFeel(), DewPoint(), WindChill()]
    },
    'deterministic': True
}

models_configs += [
    pattern_regression,
    pattern_regression_daily
]

from models.LGBM import LGBMrobust, LGBMsimple
from preprocessing.advanced_transforms import LGBM_demand_features, LGBM_impute_nan_demand
from preprocessing.advanced_transforms import LGBM_impute_nan_weather, LGBM_weather_features
from preprocessing.advanced_transforms import  LGBM_prepare_test_dfs

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

lgbm_simple = {
    'name': 'LGBMsimple',
    'model': LGBMsimple(lgb_params = lgb_params),
    'preprocessing': {
        'demand': [Logarithm(), LGBM_impute_nan_demand(), LGBM_demand_features(no_last_week=1)],
        'weather': [LGBM_impute_nan_weather(), LGBM_weather_features()],
        'prepare_test_dfs': [LGBM_prepare_test_dfs()]
    },
    'deterministic': False
}
lgbm_robust = {
    'name': 'LGBMrobust',
    'model': LGBMrobust(lgb_params = lgb_params),
    'preprocessing': {
        'demand': [Logarithm(), LGBM_impute_nan_demand(), LGBM_demand_features(no_last_week=1)],
        'weather': [LGBM_impute_nan_weather(), LGBM_weather_features()],
        'prepare_test_dfs': [LGBM_prepare_test_dfs()]
    },
    'deterministic': False
}
lgbm_simple_with_last_week = {
    'name': 'LGBMsimple_with_last week',
    'model': LGBMsimple(lgb_params = lgb_params),
    'preprocessing': {
        'demand': [Logarithm(), LGBM_impute_nan_demand(), LGBM_demand_features(no_last_week=0)],
        'weather': [LGBM_impute_nan_weather(), LGBM_weather_features()],
        'prepare_test_dfs': [LGBM_prepare_test_dfs()]
    },
    'deterministic': False
}

models_configs += [
    lgbm_simple,
    lgbm_robust,
    lgbm_simple_with_last_week
]

from models.LGBM import XGBMsimple

xgb_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.02,
    'max_depth': 6,
    'subsample': 0.8,
    'objective':'reg:squarederror',
    'min_child_weight':10,
    'silent':1
}

xgbm_simple = {
    'name': 'XGBMsimple',
    'model': XGBMsimple(xgb_params = xgb_params),
    'preprocessing': {
        'demand': [Logarithm(), LGBM_impute_nan_demand(), LGBM_demand_features(no_last_week=0)],
        'weather': [LGBM_impute_nan_weather(), LGBM_weather_features()],
        'prepare_test_dfs': [LGBM_prepare_test_dfs()]
    },
    'deterministic': False
}

models_configs += [
    xgbm_simple
]

from models.TSMix import TSMix

tsmix = {
    'name': 'TSMix',
    'model': TSMix(train_epochs=50, dropout=0.8),
    'preprocessing': {
        'demand': [Logarithm(), LGBM_impute_nan_demand()],
        'weather': []
    },
    'deterministic': False
}

models_configs += [
    tsmix
]

from models.wavenet import WaveNetModel, WaveNet_prepare_test_dfs, cfg

wavenet = {
    'name': 'WaveNet',
    'model': WaveNetModel(cfg),
    'preprocessing': {
        'demand': [],
        'weather': [],
        'prepare_test_dfs': [WaveNet_prepare_test_dfs()]
    },
    'deterministic': False
}

models_configs += [
    wavenet
]
cfg['device'] = 'cuda' # if you have a compatible NVIDIA GPU
cfg['device'] = 'mps:0' # if you have Metal acceleration on your Mac (https://developer.apple.com/metal/pytorch/)
cfg['device'] = 'cpu' # for every other machine without GPU acceleration
# Now, we can run the training of all these models and see how they perform
wfe.curr_phase='train'
wfe.n_train_seeds = 1
for config in models_configs:
    wfe.add_model(config)
