from eval.evaluator import WaterFuturesEvaluator
from eval.dashboard import run_dashboard
import models
from models.benchmarks import average_week, previous_week
from models.autoregressives import autoreg_no_preprocess, autoreg_log, autoreg_log_norm
from models.prototype_based import prototype_3, prototype_5, prototype_7, prototype_fe, prototype_fe_subset 
from models.LGBM import lgbm_simple


wfe = WaterFuturesEvaluator()

wfe.add_model(previous_week)
wfe.add_model(average_week)
wfe.add_model(autoreg_no_preprocess)
wfe.add_model(autoreg_log)
wfe.add_model(lgbm_simple)
wfe.add_model(prototype_3)
wfe.add_model(prototype_5)
wfe.add_model(prototype_7)
wfe.add_model(prototype_fe)
wfe.add_model(prototype_fe_subset)

run_dashboard(wfe)
