from .client_code import train_model, get_data
from ..Utility import BaseClient
from ..Models import MajorityVotingEnsemble
import numpy as np

class AggregatedTreesClient(BaseClient):

    def train(self, model, time):
        model = train_model(model, time)
        return model
    
    def evaluate(self, model: MajorityVotingEnsemble):
        '''
        Evaluate the model on the client's data.
        
        Args:
            model (MajorityVotingEnsemble): The model to evaluate.
        
        Returns:
            tp (int): Number of true positives.
            fp (int): Number of false positives.
            fn (int): Number of false negatives.
            tn (int): Number of true negatives.
        '''
        X, y = get_data()
        y_pred = model.predict(X)
        y_pred = np.array(y_pred)
        tp = np.sum((y == 1) & (y_pred == 1)).tolist()
        fp = np.sum((y == 0) & (y_pred == 1)).tolist()
        fn = np.sum((y == 1) & (y_pred == 0)).tolist()
        tn = np.sum((y == 0) & (y_pred == 0)).tolist()
        return tp, fp, fn, tn