from wflib.evaluator import WaterFuturesEvaluator as WFE

from baselines_impl import RollingAverageWeek, ExpWeightedRollingWeek, AutoRollingAverageWeek

from preprocessing.impute_and_fill import FillZero, FillAvgWeek

wfe = WFE()

wfe.add_model_configuration({
    'name': 'prev_week_fill_zeros',
    'model': RollingAverageWeek(1),
    'preprocessing': {
        'target': [FillZero()],
        'exogenous': []
    },
    'deterministic': True
})

wfe.add_model_configuration({
    'name': 'prev_week_fill_avg',
    'model': RollingAverageWeek(1),
    'preprocessing': {
        'target': [FillAvgWeek()],
        'exogenous': []
    },
    'deterministic': True
})

wfe.add_model_configuration({
    'name': 'rolling_average_2',
    'model': RollingAverageWeek(2),
    'preprocessing': {
        'target': [FillAvgWeek()],
        'exogenous': []
    },
    'deterministic': True
})

wfe.add_model_configuration({
    'name': 'rolling_average_4',
    'model': RollingAverageWeek(4),
    'preprocessing': {
        'target': [FillAvgWeek()],
        'exogenous': []
    },
    'deterministic': True
})

wfe.add_model_configuration({
    'name': 'rolling_average_8',
    'model': RollingAverageWeek(8),
    'preprocessing': {
        'target': [FillAvgWeek()],
        'exogenous': []
    },
    'deterministic': True
})

wfe.add_model_configuration({
    'name': 'avg_week',
    'model': RollingAverageWeek(None),
    'preprocessing': {
        'target': [],
        'exogenous': []
    },
    'deterministic': True
})

wfe.add_model_configuration({
    'name': 'rolling_average_2_exp',
    'model': ExpWeightedRollingWeek(2),
    'preprocessing': {
        'target': [FillAvgWeek()],
        'exogenous': []
    },
    'deterministic': True
})

wfe.add_model_configuration({
    'name': 'rolling_average_4_exp',
    'model': ExpWeightedRollingWeek(4),
    'preprocessing': {
        'target': [FillAvgWeek()],
        'exogenous': []
    },
    'deterministic': True
})

wfe.add_model_configuration({
    'name': 'rolling_average_8_exp',
    'model': ExpWeightedRollingWeek(8),
    'preprocessing': {
        'target': [FillAvgWeek()],
        'exogenous': []
    },
    'deterministic': True
})

wfe.add_model_configuration({
    'name': 'rolling_average_auto',
    'model': AutoRollingAverageWeek(),
    'preprocessing': {
        'target': [FillAvgWeek()],
        'exogenous': [],
        'prediction_requires_extended_dfs': False
    },
    'deterministic': False,
    'n_eval_runs': 20
})
