import eval
from eval.evaluator import WaterFuturesEvaluator
from eval.dashboard import run_dashboard

wfe = WaterFuturesEvaluator()
wfe.next_iter() # load the truth data for the plots

run_dashboard(wfe)