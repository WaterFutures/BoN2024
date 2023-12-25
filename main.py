from eval_framework import wf_evaluator as ef
from Models.benchmarks import PreviousWeek, AverageWeek
from Models.autoregressives import AutoRegressive, Arima
from Models.LGBM import LGBMsimple

RETRAIN = False

wfe = ef.WaterFuturesEvaluator()

previous_week = PreviousWeek()
average_week = AverageWeek()
#arima = Arima()
autoregressive = AutoRegressive(lags=24*14)

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
lgbm = LGBMsimple(lgb_params=lgb_params)

wfe.add_model(previous_week, force=RETRAIN)
wfe.add_model(average_week, force=RETRAIN)
#wfe.add_model(arima, force=RETRAIN)
wfe.add_model(autoregressive, force=RETRAIN)
wfe.add_model(lgbm, force=RETRAIN)

wfe.run_dashboard()
