from .client_code import finetune_xgb, get_data
from ..Utility import BaseClient

class CyclicXGBClient(BaseClient):
    
    def __init__(self):
        super().__init__()
        self.config = None

    def train(self, model, num_rounds):
        X, y = get_data()
        model = finetune_xgb(model, X, y, num_rounds)
        return model

    def evaluate(self, model):
        X, y = get_data()
        if hasattr(model, '_Booster'):
            X = X[model.feature_names_in_]
        
        y_preds = model.predict(X)
        tp = sum((y == 1) & (y_preds == 1))
        fp = sum((y == 0) & (y_preds == 1))
        fn = sum((y == 1) & (y_preds == 0))
        tn = sum((y == 0) & (y_preds == 0))
        return tp, fp, fn, tn