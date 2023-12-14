import eval_framework as ef

wfe = ef.WaterFuturesEvaluator()

from Models.benchmarks import PreviousWeek, AverageWeek
from Models.autoregressives import AutoRegressive

previous_week = PreviousWeek()
average_week = AverageWeek()
autoregressive = AutoRegressive(lags=24*7)

wfe.add_model(previous_week)
wfe.add_model(average_week)
wfe.add_model(autoregressive)

wfe.run_dashboard()
