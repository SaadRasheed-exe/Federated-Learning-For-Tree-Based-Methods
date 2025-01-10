from .utility import ProcessDataset, Utils
import configparser
from sklearn.metrics import f1_score
from .models import Models
from tqdm import tqdm
import logging

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

    
    def train_classifier(self, model):
        
        logging.info('-'*60)
        logging.info('Training Classifier')
        logging.info('-'*60)

        if self.val_required:
            x_train, x_test, x_val, y_train, y_test,  y_val = self.process.get_data(is_survival=False)
        else:
            x_train, x_test, y_train, y_test = self.process.get_data(is_survival=False)
            
        models = [(model, 'model')]
        
        
        # f1_score_var = 'F1 Score'
        # accuracy_var = 'Accuracy'
        
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
            # results_dict             = {}
            # params_dict              = {}
            # params_dict [model_name] = {}
            # results_dict[model_name] = {}
            
            
            rfc = model
            try:
                rfc.fit(x_train, y_train)
            except Exception as e:
                logging.error('Error occurred during training: %s', e)
            
            # params_dict [model_name]['model_name'] = type(rfc)
            # params_dict [model_name].update(rfc.get_params())
            # results_dict[model_name][f1_score_var]={}
            # results_dict[model_name][accuracy_var]={}
            
            # train_f1 = f1_score(y_train, rfc.predict(x_train))
            # test_f1  = f1_score(y_test,  rfc.predict(x_test))
            # train_acc= rfc.score(x_train, y_train)
            # test_acc = rfc.score(x_test , y_test)
                    
            # for (f1, acc, split_name) in [(train_f1, train_acc, 'train'), (test_f1,  test_acc,   'test')]:
            #     results_dict[model_name][f1_score_var][split_name] = f1
            #     results_dict[model_name][accuracy_var][split_name] = acc
        
            # if self.val_required:
            #     val_f1  = f1_score(y_val,  rfc.predict(x_val))
            #     val_acc = rfc.score(x_val, y_val) 
            #     results_dict[model_name][f1_score_var]['val'] = val_f1
            #     results_dict[model_name][accuracy_var]['val'] = val_acc
                
            # self.utils.save_model(self.experiment_path, rfc, model_name, is_survival=False)
            # del rfc
            
            # self.utils.save_results(self.experiment_path, results_dict)
            # self.utils.save_model_params(self.experiment_path, params_dict)
    
        return model