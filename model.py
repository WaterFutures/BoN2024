class Model:

    def __init__(self):
        pass

    def fit(self, X_train):
        pass

    def forecast(self, X_test):
        pass

    def name (self):
        return "Model"
    
    def preprocess_data(self, train__dmas_h_q, test__dmas_h_q, train__exin_h, test__exin_h, eval__exin_h):
        return train__dmas_h_q, test__dmas_h_q, train__exin_h, test__exin_h, eval__exin_h
    
    def forecasted_dmas():
        pass

    def forecasted_dmas_idx():
        pass