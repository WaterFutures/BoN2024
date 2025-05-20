from wflib.evaluator import WaterFuturesEvaluator as WFE

from ts_mix_impl import TSMix
from preprocessing.simple_transforms import Logarithm
from preprocessing.advanced_transforms import LGBM_impute_nan_demand

wfe = WFE()

wfe.add_model_configuration({
    'name': 'ts_mix',
    'model': TSMix(train_epochs=50, dropout=0.8),
    'preprocessing': {
        'target': [Logarithm(), LGBM_impute_nan_demand()],
        'exogenous': [],
        'prediction_requires_extended_dfs': False
    },
    'deterministic': False,
    'n_eval_runs': 5
})
