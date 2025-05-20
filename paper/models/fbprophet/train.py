from wflib.evaluator import WaterFuturesEvaluator as WFE

from fbprophet_impl import Fbprophet

wfe = WFE()

wfe.add_model_configuration({
    'name': 'fb_prophet',
    'model': Fbprophet(),
    'preprocessing': {
        'target': [],
        'exogenous': []
    },
    'deterministic': True
})
