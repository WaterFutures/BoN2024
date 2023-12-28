from eval.evaluator import WaterFuturesEvaluator
from eval.dashboard import run_dashboard
import models
from models.benchmarks import average_week, previous_week
from models.autoregressives import autoreg_no_preprocess, autoreg_log, autoreg_log_norm
from models.LGBM import lgbm_simple


wfe = WaterFuturesEvaluator()

wfe.add_model(previous_week)
wfe.add_model(average_week)
wfe.add_model(autoreg_no_preprocess)
wfe.add_model(autoreg_log)
wfe.add_model(lgbm_simple)

run_dashboard(wfe)
