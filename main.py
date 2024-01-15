from eval.evaluator import WaterFuturesEvaluator
from eval.dashboard import run_dashboard
import models
from models.benchmarks import average_week, previous_week
from models.autoregressives import autoreg_no_preprocess, autoreg_log, autoreg_log_norm

from models.LGBM import lgbm_simple, lgbm_robust
from models.TSMix import tsmix
from models.pattern_regression import pattern_regression
from models.rolling_average_week import RollingAverageWeek
from models.prototype_based import prototype_3, prototype_5, prototype_7, prototype_fe, prototype_fe_subset 
from models.wavenet import wavenet_lin, wavenet_log
wfe = WaterFuturesEvaluator()

# Benchamrks
wfe.add_model(previous_week)
wfe.add_model(average_week)
wfe.add_model(autoreg_no_preprocess)
#wfe.add_model(rolling_average_week(8))

# Models
wfe.add_model(lgbm_simple)
wfe.add_model(prototype_3)
wfe.add_model(prototype_5)
wfe.add_model(prototype_7)
wfe.add_model(prototype_fe)
wfe.add_model(prototype_fe_subset)
wfe.add_model(lgbm_robust)
wfe.add_model(pattern_regression)
wfe.add_model(tsmix)
wfe.add_model(wavenet_lin)
wfe.add_model(wavenet_log)

run_dashboard(wfe)
