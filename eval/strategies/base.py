import numpy as np

class Strategy:
    def __init__(self) -> None:
        pass

    def find_best_models(self, testresults: dict) -> list:
        # testresults is a dictionary with the results for the curr iteration and 
        # the curr phase for all the selected models over all seed and test weeks

        # should return a list of names of the best models in the same order that
        # you use later on in combine_forecasts
        pass

    def combine_forecasts(self, forecasts) -> np.ndarray:
        # forecasts is a list of the forecasts for the best models in the same order
        # that you returned in find_best_models 
        pass

