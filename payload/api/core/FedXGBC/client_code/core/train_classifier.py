from .utility import ProcessDataset, Utils, finetune_xgb
import configparser
from sklearn.metrics import f1_score
from .models import Models
from tqdm import tqdm
import logging
import xgboost as xgb
import os
import joblib
import pandas as pd

class TrainClassifier:
    
    def __init__(self, experiment_path, config_file_path):
        
        self.process = ProcessDataset(experiment_path, config_file_path)
        self.experiment_path = experiment_path

        self.utils  = Utils(config_file_path)
        self.models = Models()
        
        self.config = configparser.ConfigParser()
        self.config.read(config_file_path)
        self.val_required = eval(self.config.get('BASE_SETTINGS', 'validation_set_required'))
        self.imputation_required = eval(self.config.get('FEATURES_IMPUTATION', 'imputation_required'))

        self.pretrained = eval(self.config.get('BASE_SETTINGS', 'pretrained'))

    
    def train_classifier(self):
        
        logging.info('-'*60)
        logging.info('Training Classifier')
        logging.info('-'*60)

        if self.val_required:
            x_train, x_test, x_val, y_train, y_test,  y_val = self.process.get_data(is_survival=False)

            traindata = pd.concat([x_train, y_train], axis=1)
            testdata = pd.concat([x_test, y_test], axis=1)
            valdata = pd.concat([x_val, y_val], axis=1)

            traindata.to_csv(f"./traindata.csv", index=False)
            testdata.to_csv(f"./testdata.csv", index=False)
            valdata.to_csv(f"./valdata.csv", index=False)

        else:
            x_train, x_test, y_train, y_test = self.process.get_data(is_survival=False)
            
            traindata = pd.concat([x_train, y_train], axis=1)
            testdata = pd.concat([x_test, y_test], axis=1)

            traindata.to_csv(f"./traindata.csv", index=False)
            testdata.to_csv(f"./testdata.csv", index=False)
        
        exit()
        if self.pretrained:
            pretrained_path = self.config.get('PRETRAINING', 'pretrained_models_path')
            models = self.models.get_pretrained_classifiers(pretrained_path)
        else:
            models = self.models.get_classifiers_to_train()
        
        
        
        f1_score_var = 'F1 Score'
        accuracy_var = 'Accuracy'
        
        logging.info('-'*60)
        # logging.info(f'Training {len(models)} Classification Model(s) on {x_train.shape[0]} patients and {x_train.shape[1]} features: {sorted(x_train.columns, key=len)}.')
        logging.info(('Training {} Classification Model(s) on {} patients and {} features: {}.'
            .format(len(models), x_train.shape[0], x_train.shape[1], sorted(x_train.columns, key=len))))

        logging.info('-'*60)
        for (model, model_name) in tqdm(models, desc='Training Classifiers'):
            if not self.imputation_required and str(type(model)) not in ["<class 'lightgbm.sklearn.LGBMClassifier'>",
                                                                        "<class 'xgboost.sklearn.XGBClassifier'>"]:
                logging.error(f' {model_name}({type(model)}) model does not support missing values, thus skipping this.')
                continue
            results_dict             = {}
            params_dict              = {}
            params_dict [model_name] = {}
            results_dict[model_name] = {}
            
            
            rfc = model
            try:
                if self.pretrained:


                    if str(type(rfc)) in ["<class 'xgboost.sklearn.XGBClassifier'>"]:
                        num_rounds = int(self.config.get('PRETRAINING', 'num_rounds'))
                        # multiplier = float(self.config.get('PRETRAINING', 'multiplier'))
                        # num_rounds = int(rfc.n_estimators * multiplier)
                        if hasattr(rfc, '_Booster'):
                            feature_names = model.feature_names_in_
                            x_train = x_train[feature_names]
                            x_test  = x_test[feature_names]
                            if self.val_required:
                                x_val = x_val[feature_names]
                            dtrain = xgb.DMatrix(x_train, label=y_train)
                            dtest  = xgb.DMatrix(x_test,  label=y_test)
                            rfc = finetune_xgb(rfc, dtrain, dtest, num_rounds)
                            # rfc.set_params(n_estimators=rfc.n_estimators + num_rounds)
                            # rfc.fit(x_train, y_train, eval_set=[(x_test, y_test)], xgb_model=rfc)
                        else:
                            rfc.n_estimators = num_rounds
                            rfc.fit(x_train, y_train, eval_set=[(x_test, y_test)])
                    else:
                        logging.error(f'Pretrained model {model_name}({type(rfc)}) is not supported, thus skipping this.')
                        continue
                else:
                    rfc.fit(x_train, y_train)
            except Exception as e:
                logging.error('Error occurred during training: %s', e)
                continue
            
            params_dict [model_name]['model_name'] = type(rfc)
            params_dict [model_name].update(rfc.get_params())
            results_dict[model_name][f1_score_var]={}
            results_dict[model_name][accuracy_var]={}
            
            train_f1 = f1_score(y_train, rfc.predict(x_train))
            test_f1  = f1_score(y_test,  rfc.predict(x_test))
            train_acc= rfc.score(x_train, y_train)
            test_acc = rfc.score(x_test , y_test)
                    
            for (f1, acc, split_name) in [(train_f1, train_acc, 'train'), (test_f1,  test_acc,   'test')]:
                results_dict[model_name][f1_score_var][split_name] = f1
                results_dict[model_name][accuracy_var][split_name] = acc
        
            if self.val_required:
                val_f1  = f1_score(y_val,  rfc.predict(x_val))
                val_acc = rfc.score(x_val, y_val) 
                results_dict[model_name][f1_score_var]['val'] = val_f1
                results_dict[model_name][accuracy_var]['val'] = val_acc
                
            self.utils.save_model(self.experiment_path, rfc, model_name, is_survival=False)

            if self.pretrained:
                if not os.path.exists(pretrained_path):
                    os.makedirs(pretrained_path, exist_ok=True)
                joblib.dump(model, f"{pretrained_path}/{model_name}.joblib")

            del rfc

            print(results_dict)
            print(params_dict)
            
            self.utils.save_results(self.experiment_path, results_dict)
            self.utils.save_model_params(self.experiment_path, params_dict)