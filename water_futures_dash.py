import eval
from eval.evaluator import WaterFuturesEvaluator
from eval.dashboard import run_dashboard

wfe = WaterFuturesEvaluator()
wfe.next_iter() # load the truth data for the plots
wfe.next_iter() # load the second forecast
wfe.next_iter() # load the third forecast
wfe.next_iter() # load the fourth forecast

run_dashboard(wfe)