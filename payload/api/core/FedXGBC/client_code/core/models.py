from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
import os, joblib

class Models:

    """This class returns models to be trained. \n

    Contains two separate methods for classifiers and survival models.    
    """
    
    def __init__(self):
        pass
    
    def get_survival_models_to_train(self):
        
        """Give survival models to train in the form [(model, model_name)] where \n
        model      = sksurv model \n
        model_name = the name to track the results and parameters of that model \n
        Also, try to include n_jobs=-1 into parameters for fast training.
        Plus, include verbose for tracking training of survival training because it takes time.
        --------- 
        Examples:
            models = [(RandomSurvivalForest(n_estimators=5, max_depth=5, n_jobs=-1, random_state=31), 'RSF_20N20D'), \n
                      (RandomSurvivalForest(n_estimators=10, max_depth=2, n_jobs=-1, random_state=31), 'RSF_10N10D')] \n
        --------
        Returns:
            list of tuples: each tuple contain model as first and model_name as second entry
        """
        
        models = [
                #   (RandomSurvivalForest(n_estimators=5, max_depth=20, n_jobs=-1, random_state=42),'RSF_5N20D'),
                #   (RandomSurvivalForest(n_estimators=10, max_depth=10, n_jobs=-1, random_state=42),'RSF_10N10D'),
                #   (RandomSurvivalForest(n_estimators=10, max_depth=20, n_jobs=-1, random_state=42),'RSF_10N20D'),
                  
                #   (RandomSurvivalForest(n_estimators=5, max_depth=20, n_jobs=-1, random_state=42, max_samples=0.5),'RSF_5N20D0.5MS'),
                #   (RandomSurvivalForest(n_estimators=10, max_depth=10, n_jobs=-1, random_state=42, max_samples=0.5),'RSF_10N10D0.5MS'),
                #   (RandomSurvivalForest(n_estimators=10, max_depth=20, n_jobs=-1, random_state=42, max_samples=0.5),'RSF_10N20D0.5MS'),
                  
                #   (RandomSurvivalForest(n_estimators=5, max_depth=20, n_jobs=-1, random_state=42, min_samples_leaf=10),'RSF_5N20D10Samples'),
                  (RandomSurvivalForest(n_estimators=10, max_depth=10, n_jobs=-1, random_state=42, min_samples_leaf=10),'RSF_10N10D10Samples'),
                  (RandomSurvivalForest(n_estimators=10, max_depth=20, n_jobs=-1, random_state=42, min_samples_leaf=10),'RSF_10N20D10Samples'),
                  
                #   (RandomSurvivalForest(n_estimators=5, max_depth=20, n_jobs=-1, random_state=42, min_samples_leaf=15),'RSF_5N20D15Samples'),
                  (RandomSurvivalForest(n_estimators=10, max_depth=10, n_jobs=-1, random_state=42, min_samples_leaf=15),'RSF_10N10D15Samples'),
                  (RandomSurvivalForest(n_estimators=10, max_depth=20, n_jobs=-1, random_state=42, min_samples_leaf=15),'RSF_10N20D15Samples'),
                  
                #   (RandomSurvivalForest(n_estimators=5, max_depth=20, n_jobs=-1, random_state=42, min_samples_leaf=10, max_samples=0.5),'RSF_5N20D0.5MS10samples'),
                #   (RandomSurvivalForest(n_estimators=10, max_depth=10, n_jobs=-1, random_state=42, min_samples_leaf=10, max_samples=0.5),'RSF_10N10D0.5MS10samples'),
                #   (RandomSurvivalForest(n_estimators=10, max_depth=20, n_jobs=-1, random_state=42, min_samples_leaf=10, max_samples=0.5),'RSF_10N20D0.5MS10samples'),
                  
                #   (RandomSurvivalForest(n_estimators=20, max_depth=20, n_jobs=-1, random_state=42),'RSF_20N20D'),
                  (RandomSurvivalForest(n_estimators=20, max_depth=10, n_jobs=-1, random_state=42, min_samples_leaf=10),'RSF_20N10D10Samples'),
                  (RandomSurvivalForest(n_estimators=20, max_depth=20, n_jobs=-1, random_state=42, min_samples_leaf=10),'RSF_20N20D10Samples'),
                #   (RandomSurvivalForest(n_estimators=20, max_depth=20, n_jobs=-1, random_state=42, max_samples=0.5),'RSF_20N20D0.5MS'),
                ]
        
        return models
    
    def get_classifiers_to_train(self):
        
        """
        Give classifier models to train in the form [(model, model_name)] where
        
            model      = any model from: [sklearn, XGBoost, LightGBM]

            model_name = the name to track the results and parameters of that model
        
        Ensure if missing values are supported by your model or not. Currently the check for training with
        missing values is hard coded to allow only XGBoost and LightGBM models.
        ---------
        For reproducible and fast training:
        
            For sklearn models and XGBoost; use random_state and 'n_jobs=-1' parameters.
            
            For lightgbm; use seed, deterministic and 'n_jobs=total system cores (not threads)' parameters.
            
        Examples:
        ---------         
            models =[(RandomForestClassifier(n_estimators=5, max_depth=5, n_jobs=-1, random_state=42), 'RFC_20N20D'),
                     (LGBMClassifier        (n_estimators=5, deterministic=True,n_jobs=4, seed=42 ),   'LGB_10N10D')]      
        
        Returns:
        --------
            list of tuples: each tuple contain model as first and model_name as second entry
        """
        
        models = [
                # (RandomForestClassifier(n_jobs=-1, random_state=42),'RFC_Default'),
                # (RandomForestClassifier(n_estimators=25, max_depth=10, n_jobs=-1, random_state=42),'RFC_25N10D'),
                # (RandomForestClassifier(n_estimators=25, max_depth=20, n_jobs=-1, random_state=42),'RFC_25N20D'),
                # (RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),'RFC_50N10D'),
                # (RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1, random_state=42),'RFC_50N20D'),
                # (RandomForestClassifier(n_estimators=50, max_depth=50, n_jobs=-1, random_state=42),'RFC_50N50D'),
                # (RandomForestClassifier(n_estimators=100, max_depth=50, n_jobs=-1, random_state=42),'RFC_100N50D'),
                # (RandomForestClassifier(n_estimators=100, max_depth=80, n_jobs=-1, random_state=42),'RFC_100N80D'),
                # (AdaBoostClassifier(RandomForestClassifier(n_estimators=5, max_depth=10, n_jobs=-1, random_state=42), n_estimators=5, random_state=42),'BF_5N_RF_5N10D'),
                # (AdaBoostClassifier(RandomForestClassifier(n_estimators=5, max_depth=20, n_jobs=-1, random_state=42), n_estimators=5,random_state=42),'BF_5N_RF_5N20D'),
                # (AdaBoostClassifier(RandomForestClassifier(n_estimators=10, max_depth=10, n_jobs=-1, random_state=42), n_estimators=10,random_state=42),'BF_10N_RF_10N10D'),
                # (AdaBoostClassifier(RandomForestClassifier(n_estimators=10, max_depth=20, n_jobs=-1, random_state=42), n_estimators=10,random_state=42),'BF_10N_RF_10N20D'),
                # (AdaBoostClassifier(RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42), n_estimators=10, random_state=42), 'BF_10N_RF_50N10D'),
                # (AdaBoostClassifier(RandomForestClassifier(n_estimators=1000, max_depth=10, n_jobs=-1, random_state=42), n_estimators=10, random_state=42), 'BF_10N_RF_50N1000D'),
                # (RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42), 'RFC_1000N_Default'),
                # (RandomForestClassifier(n_estimators=1000, max_depth=10, n_jobs=-1, random_state=42), 'RFC_1000N10D'),
                # (RandomForestClassifier(n_estimators=1000, max_depth=20, n_jobs=-1, random_state=42), 'RFC_1000N20D'),
                # (RandomForestClassifier(n_estimators=1000, max_depth=50, n_jobs=-1, random_state=42), 'RFC_1000N50D'),
                # (RandomForestClassifier(n_estimators=1000, max_depth=80, n_jobs=-1, random_state=42), 'RFC_1000N80D'),
                # (RandomForestClassifier(n_estimators=1000, max_depth=100, n_jobs=-1, random_state=42), 'RFC_1000N100D'),
                # (RandomForestClassifier(n_estimators=1000, max_depth=150, n_jobs=-1, random_state=42), 'RFC_1000N150D'),
                # (LGBMClassifier(deterministic=True, force_row_wise=True, seed=42, n_jobs=4),'LGBM_default'),
                # (XGBClassifier(random_state=42, n_jobs=-1), 'XGB_default'),
                (DecisionTreeClassifier(max_depth=15, random_state=42), 'DTC_15D'),
            ]        
        return models
    
    def get_pretrained_classifiers(self, path):
        """
        Get pretrained models from the given path.
        """

        models = []
        for model_name in os.listdir(path):
            if model_name.endswith('.joblib'):
                model = joblib.load(os.path.join(path, model_name))
                models.append((model, model_name.split('.')[0]))
        
        return models
