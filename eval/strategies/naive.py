from eval.strategies.base import Strategy

class Naive(Strategy):
    def __init__(self) -> None:
        super().__init__()

    def find_best_models(self, testresults):
        return [testresults.keys()[0]]

    def combine_forecasts(self, forecasts):
        return forecasts[0]