from wflib.evaluator import WaterFuturesEvaluator as WFE

from pattern_reg_impl import PatternRegression, PatternRegressionDaily

from preprocessing.simple_transforms import Logarithm
from preprocessing.weather_feature_engineering import RealFeel, DewPoint, WindChill

wfe = WFE()

wfe.add_model_configuration({
    'name': f'PatternRegression',
    'model': PatternRegression(),
    'preprocessing': {
        'target': [Logarithm()],
        'exogenous': [RealFeel(), DewPoint(), WindChill()]
    },
    'deterministic': True
})

wfe.add_model_configuration({
    'name': 'prev_week_fill_avg',
    'model': PatternRegressionDaily(),
    'preprocessing': {
        'target': [Logarithm()],
        'exogenous': [RealFeel(), DewPoint(), WindChill()]
    },
    'deterministic': True
})
