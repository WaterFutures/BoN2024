from wflib.evaluator import WaterFuturesEvaluator as WFE

from wavenet_impl import WaveNetModel

from wavenet_impl import ENCODERS # Input features, even if not in preprocessing module

# Other imports:
import yaml

wfe = WFE()

with open('config.yml') as f:
    cfg = yaml.safe_load(f)

cfg['device']='cuda'

wfe.add_model_configuration({
    'name': 'wavenet',
    'model': WaveNetModel(cfg),
    'preprocessing': {
        'target': [],
        'exogenous': [],
        'prediction_requires_extended_dfs': True
    },
    'deterministic': False,
    'n_eval_runs': 5
})
