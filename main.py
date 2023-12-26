from new_eval_framework import WaterFuturesEvaluator
from eval_dashboard import run_dashboard

import models
from models.benchmarks import average_week, previous_week
from models.autoregressives import autoreg_no_preprocess


wfe = WaterFuturesEvaluator()

wfe.add_model(previous_week)
wfe.add_model(average_week)
wfe.add_model(autoreg_no_preprocess)

run_dashboard(wfe)
