from eval.strategies.base import Strategy

class Naive(Strategy):
    def __init__(self) -> None:
        super().__init__()

    def find_best_models(self, testresults):
        # return the firts key of the dictionary
        return ['AutoRollingAverage']

    def combine_forecasts(self, forecasts):
        return forecasts['AutoRollingAverage'].groupby('Date').mean().to_numpy('float64')