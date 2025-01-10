import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from configparser import ConfigParser
import os
import copy
import joblib
import json
import logging
import shutil

class Utils:
    
    """This class contains all the utility functions required for all other classes/files."""
    
    
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config = ConfigParser()
        self.config.read(config_file_path)
        self.ref_date_clmn         = self.config.get('COLUMN_NAMES', 'ref_date_clmn')
        self.thresh_date_clmn      = self.config.get('COLUMN_NAMES', 'thresh_date_clmn')
        self.diagnosis_date_clmn   = self.config.get('COLUMN_NAMES', 'diagnosis_date_clmn')
        self.latest_encounter_clmn = self.config.get('COLUMN_NAMES', 'latest_encounter_clmn')
        self.time_column_name      = self.config.get('COLUMN_NAMES', 'time_column_name')
        self.additional_clmns_save = eval(self.config.get('SPLITS_SAVE_CONFIG', 'additional_clmns_needed'))
        self.how_many_months_model = self.config.get('BASE_SETTINGS', 'how_many_months_model')
    
    

    def configure_logging(self, log_directory: str, printing_required: bool) -> None:
        """
        Configure logging settings.

        This method sets up logging for the application. It creates a log directory
        if it doesn't exist, configures logging to write both to a log file and optionally to
        the console based on user preference, and sets the logging level to INFO.

        Call this method only once in main file.

        Parameters:
            log_directory (str): The directory where logs will be stored.
            printing_required (bool): A flag indicating whether logs should be printed to console.
        """
        # Generate timestamp
        _, timestamp = log_directory.split("trial ")
        # Create 'log' directory if it doesn't exist
        os.makedirs(log_directory := os.path.join(log_directory, 'Logs'), exist_ok=True)

        # Set up logging
        log_file = os.path.join(log_directory, f'modelling_logs_{timestamp}.log')  # Set log file path with timestamp

        # Create logger
        logger = logging.getLogger()  # Get logger object
        logger.setLevel(logging.INFO)  # Set logging level to INFO

        # Create file handler and set level to info
        file_handler = logging.FileHandler(log_file)  # Create file handler for logging
        file_handler.setLevel(logging.INFO)  # Set file handler's logging level to INFO

        # Create formatter and add to handlers
        file_formatter = logging.Formatter('%(levelname)s - %(message)s')  # Define format for file output
        file_handler.setFormatter(file_formatter)  # Set formatter for file handler

        # Add file handler to the logger
        logger.addHandler(file_handler)  # Add file handler to the logger

        if printing_required:
            # Create console handler and set level to info
            console_handler = logging.StreamHandler()  # Create console handler for logging
            console_handler.setLevel(logging.INFO)  # Set console handler's logging level to INFO

            # Create formatter for console handler
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')  # Custom format for console output
            console_handler.setFormatter(console_formatter)  # Set formatter for console handler

            # Add console handler to the logger
            logger.addHandler(console_handler)  # Add console handler to the logger

    def log_details(self, df, columns):
        """Utility function to record dataset details just before training."""
        logging.info('------------------- DATASET DETAILS ----------------')
        
        for column in columns:
            logging.info(f"{column} has following value counts.")
            value_counts = df[column].value_counts()
            for value, count in value_counts.items():
                logging.info(f"    {int(value)}: {count} entries")
        
        logging.info('-'*60)
        return df
    
    def log_null_stats(self, df, column_list):
        logging.info("------------------COUNT OF NULL VALUES-------------------")
        
        # Count null values for each class
        class_null_counts = df.groupby('is_diagnosed')[column_list].apply(lambda x: x.isnull().sum())
        
        for col in column_list:
            null_count_class_0 = class_null_counts.loc[0, col]
            null_count_class_1 = class_null_counts.loc[1, col]
            total_null_count = null_count_class_0 + null_count_class_1
            
            logging.info(f"There are {total_null_count} null values for {col} column. {null_count_class_0} in negative class and {null_count_class_1} in positive class.")
        
        logging.info('-'*60)

    def load_data(self, is_diagnosed, filename, filepath):
        
        if not filepath:
            raise ValueError(f'{filepath} not allowed as file path. Kindly correct it.')
        try:
            self.check_filepath(filepath)
        except FileNotFoundError as e:
            logging.critical(e)
        dates_to_parse = [self.ref_date_clmn]
        if is_diagnosed:
            dates_to_parse.append(self.diagnosis_date_clmn)
        else:
            dates_to_parse.append(self.latest_encounter_clmn)
        
        disease_name = self.config.get('BASE_SETTINGS', 'modelling_which_disease')
        df = pd.read_csv(f'{filepath}/{disease_name}/{filename}.csv', parse_dates=dates_to_parse, low_memory=False)
        return df
    
    def load_model(self):
        
        filename = self.config.get('FILE_SETTINGS', 'model_filename')
        filepath = self.config.get('FILE_SETTINGS', 'model_filepath')
        if not filepath:
            raise ValueError(f'{filepath} not allowed as file path. Kindly correct it.')
        try:
            self.check_filepath(filepath)
        except FileNotFoundError as e:
            logging.critical(e)
            
        model = joblib.load(f'{filepath}/{filename}.joblib')
        
        return model
        
    def rename_column(self, df, orignal_column = None, 
                                new_column     = None):
        
        if orignal_column is None:
            orignal_column = self.diagnosis_date_clmn
        if new_column is None:
            new_column = self.latest_encounter_clmn

        if orignal_column not in df.columns:
            logging.error(ValueError(f'{orignal_column} not in {df.columns}'))
            return df
        
        df.rename(columns={orignal_column: new_column}, inplace=True)
        
        return df
    
    def check_filepath(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file path {filepath} does not exist.")
        
    def get_and_make_experiment_path(self, time):
        self.experiment_path = os.path.join(
            "models",
            self.config.get('BASE_SETTINGS', 'modelling_which_disease'),
            self.config.get('FILE_SETTINGS', 'prefix_to_experiment_path') + "trial " + str(time).split('.')[0].replace(':','_'))
        os.makedirs(self.experiment_path)
        return self.experiment_path
    
    def save_config_file(self, experiment_path):
        model_type = eval(self.config.get('BASE_SETTINGS', 'model_type'))
        if model_type == 0:
            config_filename = 'classification_config.ini'
        elif model_type == 1:
            config_filename = 'survival_config.ini'
        else:
            config_filename = 'classification_and_survival.ini'

        destination_file_path = os.path.join(experiment_path, config_filename)

        # Copy the file to the desired location
        shutil.copyfile(self.config_file_path, destination_file_path)
            
    def save_splits(self, experiment_path, shortlisted_features, train, test, val=None):
        
        self.splits_path = os.path.join(experiment_path, 'splits')
        if not os.path.exists(self.splits_path):
            os.mkdir(self.splits_path)
            time_columns = [self.diagnosis_date_clmn, self.thresh_date_clmn, self.ref_date_clmn]
            save_columns = copy.deepcopy(self.additional_clmns_save)
            save_columns.extend(time_columns)
            save_columns.extend(shortlisted_features)
            save_columns.append(self.time_column_name)
            save_columns.append('is_diagnosed')
            for (x, split_name) in [ (train, 'train'), (val, 'val'), (test,  'test')]:
                if isinstance(x, pd.DataFrame): # Check if Val None
                    x[save_columns].to_csv(f'{self.splits_path}/{split_name}.csv', index=False)
                
    def save_model(self, experiment_path, model, model_name, is_survival):
        """Saves model to file if config setting for this is true."""
        if eval(self.config.get('BASE_SETTINGS', 'save_model')):
            model_path = os.path.join(experiment_path, 'models')
            if is_survival:
                model_name += '_survival'
            else:
                model_name += '_classification'
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            joblib.dump(model, f"{model_path}/{self.how_many_months_model}month_{model_name}.joblib")
            
    def save_model_params(self, experiment_path, params_dict):
        """Save model parameters of trained models and reuse same file if params file already exists."""
        params_df = pd.DataFrame.from_dict(params_dict)
        params_path = os.path.join(experiment_path, 'Parameters')
        excel_name  = f'{params_path}/{self.how_many_months_model}month_params_{experiment_path[experiment_path.find("trial"):]}'
        
        if os.path.exists(params_path):
            df_already = pd.read_excel(f'{excel_name}.xlsx', index_col=0)
            if df_already.shape[0]>params_df.shape[0]:
                left_df, right_df = df_already, params_df
            else:
                left_df, right_df = params_df, df_already
            missing_indices = left_df.index.difference(right_df.index)
            params_df = left_df.merge(right_df, how='outer', left_index=True, right_index=True)
            params_df.loc[missing_indices, right_df.columns] = params_df.loc[missing_indices, right_df.columns].fillna('Not Exists')
        else:
            os.mkdir(params_path)
        params_df.to_excel(f'{excel_name}.xlsx', index=True)
        
    def save_results(self, experiment_path, results_dict):
        """Save results after model training. Reuse the same file if some results already exist."""
        model_names = list(results_dict.keys())
        result_types= list(results_dict[model_names[0]].keys()) # Because atleast one model would be there
        split_types = list(results_dict[model_names[0]][result_types[0]].keys())
        
        columns_tuples = [(name, type) for name in model_names for type in result_types]
        columns = pd.MultiIndex.from_tuples(columns_tuples)
        # split_types = ['train', 'val', 'test']
        data  = [[results_dict[model_name][result_type][split_type] for model_name in results_dict for result_type in results_dict[model_name]] for split_type in split_types]
        df    = pd.DataFrame(data, index=split_types, columns=columns)
        results_path = os.path.join(experiment_path, 'Results')
        excel_name     = f'{results_path}/{self.how_many_months_model}month_results_{experiment_path[experiment_path.find("trial"):]}'
        
        if os.path.exists(results_path):
            df_already = pd.read_excel(f'{excel_name}.xlsx', index_col=0, header=[0, 1])
            df         = df_already.merge(df, how='inner', left_index=True, right_index=True)
        else:
            os.mkdir(results_path)
        df.to_excel(f'{excel_name}.xlsx', index=True)
        
    def save_imputations(self, experiment_path, imputation_dict):
        """Save imputations done on the combined dataset."""
        imputation_df   = pd.DataFrame.from_dict(imputation_dict)
        imputation_path = os.path.join(experiment_path, 'Imputations')
        csv_name        = f'{imputation_path}/{self.how_many_months_model}month_imputations_{experiment_path[experiment_path.find("trial"):]}'

        if os.path.exists(imputation_path):
            df_already = pd.read_csv(f'{csv_name}.csv', index_col=0)
            if len(set(df_already.index).intersection(set(imputation_df.index)))!=df_already.shape[0]:
                raise ValueError('Imputations are not being saved correctly. Kindly review.')
        else:
            os.mkdir(imputation_path)
        imputation_df.to_csv(f'{csv_name}.csv', index=True)        
        
    def get_config_value(self, config, section, key):
        """ 
        Get a value from the config, converting it to int if possible, 
        or to None if it's an empty string, otherwise return as a string.
        """
        value = config.get(section, key)
        if value == '':
            return None
        try:
            return int(value)
        except ValueError:
            return value
        
    def plot_time(self, combined_df, 
                        save_fig                 = None,
                        save_fig_name            = None,
                        save_fig_path            = None,
                        time_column_name         = None,
                        diagnosis_flag_clmn_name = 'is_diagnosed',
                        lower_bound_diagnosed    = None,
                        upper_bound_normal       = None,
                        bins                     = 60):
        """Plots time of the combined dataframe after processing and before splitting."""

        if save_fig is None:
            save_fig = eval(self.config.get('BASE_SETTINGS', 'save_timeplot'))
        if time_column_name is None:
            time_column_name = self.time_column_name
        if lower_bound_diagnosed is None:
            lower_bound_diagnosed = eval(self.config.get('FEATURE_RANGES', 'time_in_days_range'))[0]
        if upper_bound_normal is None:
            upper_bound_normal = eval(self.config.get('FEATURE_RANGES', 'time_in_days_range'))[1]
        if save_fig_name is None:
            save_fig_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        elif 'trial' in save_fig_name:
            save_fig_name = save_fig_name[save_fig_name.find('trial'):]
        if not save_fig_path:
            save_fig_path = 'not_in_right_directory'

        plt.figure(figsize=(25,8))
        plt.title('Time Histogram')
        plt.xlabel('Time in days')
        plt.ylabel('Count')
        plt.hist(combined_df[(combined_df[diagnosis_flag_clmn_name]==1) &(combined_df[time_column_name]>=lower_bound_diagnosed) & (combined_df[time_column_name]<=upper_bound_normal)][time_column_name],bins=bins, label='diagnosed', alpha=0.6, color='b')
        plt.hist(combined_df[(combined_df[diagnosis_flag_clmn_name]==0) &(combined_df[time_column_name]>=lower_bound_diagnosed) & (combined_df[time_column_name]<=upper_bound_normal)][time_column_name],bins=bins, label='normal', alpha=0.5, color='r')
        plt.legend()
        if save_fig:
            save_fig_path = os.path.join(save_fig_path, 'Plots')
            if not os.path.exists(save_fig_path):
                os.mkdir(save_fig_path)
                plt.savefig(f'{save_fig_path}/Time_Plot_LB{lower_bound_diagnosed}_UB{upper_bound_normal}_{save_fig_name}.png')
    
    def check_unique_patients(self, df, practice_clmn  = 'Practice',
                                        patientid_clmn = 'PatientID'):
        max_occurrence = df.groupby(by=[practice_clmn, patientid_clmn]).size().max()
        if max_occurrence>1:
            occurrences          = df.groupby(by=[practice_clmn, patientid_clmn]).size().to_dict()
            multiple_occurrences = [patient for patient, occ in occurrences.items() if occ>1]
            logging.warning(ValueError(f'Total {len(multiple_occurrences)} patients: {multiple_occurrences[:10]}... occur multiple times.'))
            df = self._drop_duplicate_patients(df)
            
    def _drop_duplicate_patients(self, df):
        
        orignal_rows = df.shape[0]
        df.drop_duplicates(subset=['Practice', 'PatientID'], inplace=True, keep='last')
        logging.info(f'Dropping {orignal_rows-df.shape[0]} duplicate patients.')
        self.check_unique_patients(df) # sanity check
        
        return df
    
    def check_class_imbalance(self, df, print_counts=True):
        """Class imbalance checker method."""
        counts = df.value_counts()
        if print_counts:
            logging.info(f'Positive Count: {counts[1]}')
            logging.info(f'Negative Count: {counts[0]}')
        logging.info(f'Positive Ratio: {round(counts[1]*100/df.shape[0], 3)}')
        
    def print_shapes(self, diagnosed_df, normal_df):
        """Print shapes of diagnosed and normal dataframe."""
        logging.info(f'Diagnosed shape: {diagnosed_df.shape}')
        logging.info(f'Normal    shape: {normal_df.shape}')

    def balance_dataset(self, combined_df):
        """Undersamples normal dataset."""
        count_is_diag_1   = combined_df[combined_df['is_diagnosed']==1].shape[0]
        count_is_diag_0   = combined_df[combined_df['is_diagnosed']==0].shape[0]
        try:
            sampled_is_diag_0 = combined_df[combined_df['is_diagnosed'] == 0].sample(n=count_is_diag_1, random_state=42)
            logging.info('Balancing dataset by undersampling normal patients.')
            logging.info(f'Dropping {count_is_diag_0 - count_is_diag_1} patients from Negative class.')
        except ValueError:
            logging.error('Dataset can not be balanced, because normal dataset is less than diagnosed.')
            return combined_df
        balanced_df = pd.concat([sampled_is_diag_0, combined_df[combined_df['is_diagnosed'] == 1]])
        
        return balanced_df

    def _process_diagnosis_codes_duration(self, shortlisted_features):
        
        """Compares the primary codes of the chronic diseases for duration columns in
        shortlisted features. Returns shortlisted_features minus the duration_of_code
        column where code is not in chronic diseases list.
        """
        try:
            with open('./chronic_diseases_list.json', 'r') as f:
                chronic_diseases_list = json.load(f)
        except Exception as e:
            logging.error(f"Error reading chronic diseases file: {e}")
            return shortlisted_features
            
        shortlisted_features           = copy.deepcopy(shortlisted_features)
        chronic_diseases_primary_codes = list({x[:3] for x in chronic_diseases_list})
        duration_columns               = [x for x in shortlisted_features if 'duration' in x]
        duration_col_codes             = [x[12:] for x in duration_columns]
        duration_col_codes_chronic     = list(set(duration_col_codes).intersection(set(chronic_diseases_primary_codes)))
        duration_columns_chronic       = [x for x in duration_columns if x[12:] in duration_col_codes_chronic]

        for feature in shortlisted_features.copy():
            if 'duration' in feature and feature not in duration_columns_chronic:
                shortlisted_features.remove(feature)
        
        return shortlisted_features
    
    def get_shortlisted_features(self):
        
        shortlisted_features  = eval(self.config.get('SHORTLISTED_FEATURES', 'shortlisted_features'))
        
        if shortlisted_features is None:
            diag_columns    = self.load_data(is_diagnosed=True, 
                                             filename= self.config.get('FILE_SETTINGS', 'diagnosed_filename'), 
                                             filepath= self.config.get('FILE_SETTINGS', 'dataset_filepath')).columns
            
            normal_columns  = self.load_data(is_diagnosed=False,
                                             filename= self.config.get('FILE_SETTINGS', 'normal_filename'), 
                                             filepath= self.config.get('FILE_SETTINGS', 'dataset_filepath')).columns
            all_columns     = set(diag_columns).intersection(set(normal_columns))
            exclude_columns = eval(self.config.get('EXCLUSION', 'exclude_columns'))
            shortlisted_features = list(set(all_columns).difference(set(exclude_columns)))
        
        shortlisted_features = self._process_diagnosis_codes_duration(shortlisted_features)
            
        return list(set(shortlisted_features))