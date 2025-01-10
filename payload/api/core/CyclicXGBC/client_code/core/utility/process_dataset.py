import pandas as pd
import numpy as np
from datetime import datetime
import configparser
import copy
from .utils import Utils
from sklearn.model_selection import train_test_split
import os
import logging

class ProcessDataset:
    """
    This class processes both diagnosed and normal datasets for training of models. 
    Paths for datasets, the null column handling, balancing, imputations etc.
    should be configured in the corresponding config.ini file.
    """
    
    def __init__(self, experiment_path, config_file_path):
        
        self.utils=Utils(config_file_path)
        self.config = configparser.ConfigParser()
        self.config.read(config_file_path)
        config = self.config

        self.null_col_to_drop         = eval(config.get('NULL_COLUMNS', 'null_col_to_drop'))
        self.ref_date_clmn            = config.get('COLUMN_NAMES', 'ref_date_clmn')
        self.diagnosis_date_clmn      = config.get('COLUMN_NAMES', 'diagnosis_date_clmn')
        self.latest_encounter_clmn    = config.get('COLUMN_NAMES', 'latest_encounter_clmn')
        self.time_column_name         = config.get('COLUMN_NAMES', 'time_column_name')
        self.val_required             = eval(config.get('BASE_SETTINGS', 'validation_set_required'))
        self.imputation_required      = eval(config.get('FEATURES_IMPUTATION', 'imputation_required'))
        self.duration_col_upper_bound = eval(config.get('BASE_SETTINGS', 'duration_col_upper_bound'))


        self.experiment_path = experiment_path
        self.shortlisted_features = self.utils.get_shortlisted_features()
        self.imputation_dict               = {}
        self.imputation_dict['Imputation'] = {}
        
        # Try except blocks added, if by chances the configured wrong, they can be handled.
        try:  
            self.lab_ranges = eval(self.config.get('FEATURE_RANGES', 'lab_ranges'))
        except:
            self.lab_ranges =  {'total rbc' : (3.8, 5.8),
                                'total wbc' : (3.4, 10.8),
                                'hematocrit': (34, 50),
                                'platelet'  : (130, 450),
                                'hemoglobin': (11, 18)}
        try:  
            self.non_lab_ranges = eval(self.config.get('FEATURE_RANGES', 'non_lab_ranges'))
        except:
            self.non_lab_ranges =  {'age':      (18, 120),
                                    'height':    (150, 213), # Average height for 18 years is 150 cm and 213 cm (7 feet) is considered the upper limit.
                                    'weight':    (35, 300),
                                    'diastolic': (40, 120),
                                    'systolic':  (70, 220)}
            
        try:
            self.exclude_columns = eval(self.config.get('EXCLUSION', 'exclude_columns'))
        except:
            self.exclude_columns = ['mean_arterial_pressure', 'respiration', 'temperature', 'family_I11', 
                                    'weight', 'height', 'pulse']
            
        try:
            self.mapping = eval(self.config.get('MAPPING', 'mappings'))
        except:
            self.mapping = None
            
        try:
            self.imputation_required = eval(config.get('FEATURES_IMPUTATION', 'imputation_required'))
        except:
            self.imputation_required = True
        try:
            self.impute_non_lab_null_with= eval(self.config.get('FEATURES_IMPUTATION', 'impute_non_lab_null_with'))
        except:
            self.impute_non_lab_null_with=0    
        #--------------------------------------------------------------------------------------------------------------------------#
        try:
            if self.imputation_required: 
                self.specific_imputations = eval(self.config.get('FEATURES_IMPUTATION', 'specific_imputations'))
        except:
            self.impute_non_lab_null_with=0
        #--------------------------------------------------------------------------------------------------------------------------#
        try:
            self.grouping_required = eval(config['FEATUREGROUPS']['grouping_required'])
            if self.grouping_required:
                self.features_to_group = eval(config['FEATUREGROUPS']['features_to_group'])
        except:
            self.grouping_required = False           


    def drop_columns(self, diagnosed_df, normal_df):
        
        """Drop columns which are not common in diagnosed and normal dataframes.

        Returns:
            Pandas dataframes
        """
        
        more_in_diagnosed = diagnosed_df.shape[1] > normal_df.shape[1]
        col_diff          = list(set(diagnosed_df.columns).symmetric_difference(set(normal_df.columns)))
        col_diff          = [col for col in col_diff if col not in [self.diagnosis_date_clmn, self.latest_encounter_clmn]]
        if col_diff:
            logging.warning(f'There are {len(col_diff)} columns: {sorted(col_diff)} different in diagnosed and normal datasets which are being dropped.')
            
        if more_in_diagnosed:
            diagnosed_df.drop(columns=col_diff, axis=0, inplace=True) # Drop whichever has larger
        else:
            normal_df.drop(columns=col_diff, axis=0, inplace=True)
            
        return diagnosed_df, normal_df

    def concat_dataset(self, diagnosed_df, normal_df):
        
        """Combines diagnosed and normal dataframes.

        Returns:
            Pandas dataframe: combined dataframe 
        """
        
        combined    = pd.concat([diagnosed_df, normal_df], axis=0)
        combined.reset_index(inplace=True, drop=True)
        
        return combined
    
    def get_time_column(self, df, ref_date_column   = None, 
                                  after_date_column = None,
                                  time_column_name  = None):
        
        """
        Get time column based on difference between reference date and
        latest encounter/diagnosis date (based on dataset).
        
        Returns:
            df with time column added.
        """

        if ref_date_column is None:
            ref_date_column = self.ref_date_clmn
        if after_date_column is None:
            after_date_column = self.diagnosis_date_clmn
        if time_column_name is None:
            time_column_name = self.time_column_name

        
        df[time_column_name] = df.apply(lambda x: (x[after_date_column]-x[ref_date_column]).days, axis=1)
        
        return df
    
    def _drop_null_col(self, df, column_to_drop):
        """
        Drops null rows based on a specific column.

        Returns:
            df with null rows for a specific column dropped.
        """
        # Count the number of patients from each class before dropping null rows
        class_counts_before = df['is_diagnosed'].value_counts()

        df = df.dropna(subset=[column_to_drop])

        # Count the number of patients from each class after dropping null rows
        class_counts_after = df['is_diagnosed'].value_counts()

        # Calculate the number of dropped patients for each class
        dropped_counts = class_counts_before - class_counts_after

        # Log the information about dropped patients for each class
        logging.info(f"Dropping {dropped_counts.sum()} patients based on null {column_to_drop} column. {dropped_counts[0]} from negative class and {dropped_counts[1]} from positive class.")

        return df

    def _convert_to_int(self, df, column_name, print_int_stats):
        """Converts designated column to int datatype."""
        df[column_name] = df[column_name].apply(lambda x: round(x,0))
        try:
            df[column_name] = df[column_name].astype(int)
            if print_int_stats:
                logging.info(f'Converting {column_name} column to int datatype.')
        except Exception as e:
            logging.error(f'Error {e}, thus not converting {column_name} column to int datatype.')
        return df
    
    def _keep_non_null_within_threshold(self, df, column_name, column_range):
        """For a specific column, keep all non-null values within the specified range."""
        
        min_value = column_range[0]
        max_value = column_range[1]
        
        original_rows = df.shape[0]
        
        # Filter the dataframe based on the column range or null values
        df_filtered = df[(df[column_name] >= min_value) & (df[column_name] <= max_value) | (df[column_name].isna())]
        
        # Calculate the number of dropped patients due to null values
        null_counts = original_rows - df_filtered.shape[0]
        
        # Calculate the number of dropped patients for each class due to filtering
        dropped_counts = df['is_diagnosed'].value_counts() - df_filtered['is_diagnosed'].value_counts()

        # Log the information about dropped patients for each class
        logging.info(f'Dropping {null_counts} patients for non-null {column_name} column values not within {column_range} range. '
                    f'{abs(dropped_counts[0])} from negative class and {abs(dropped_counts[1])} from positive class.')
        
        return df_filtered

    def _map_column(self, df, column_name, mapping, print_map_stats):
        
        """Map a specific column to a pre-defined mapping. Mappings should be defined in config file.
        """
        
        df[column_name] = df[column_name].replace(mapping)
        if print_map_stats:
            logging.info(f'Mapping {column_name} column.')
        return df
    
    def _fill_null(self, df, column_list):
        
        """Fill null values in specific column with a specific value. This value is configurable.
        """
        logging.info("----------------------IMPUTATIONS------------------------")
        for col in column_list:
            # Check if the column has a specific imputation defined
            if self.specific_imputations and col in self.specific_imputations.keys():
                # Apply the specific imputation to only the rows with NA in the current column
                df[col] = df[col].fillna(self.specific_imputations[col])
                # Update the imputation dictionary for this specific imputation
                logging.info(f'Null values for {col} column are being filled with {self.specific_imputations[col]} value.')
                self.imputation_dict['Imputation'][col] = self.specific_imputations[col]
            else:
                fill_with= self.impute_non_lab_null_with
                # Count the number of nulls in the current column
                number_of_nulls = sum(df[col].isna())
                if number_of_nulls:
                    logging.info(f'Null values for {col} column are being filled with {fill_with} value.')
                    # Update the imputation dictionary for this column with the generic fill_with value
                    self.imputation_dict['Imputation'][col] = fill_with

        # After handling specific imputations individually, fill remaining NA values in the dataframe
        df[column_list] = df[column_list].fillna(fill_with)

        # Save the imputation dictionary after processing all columns
        self.utils.save_imputations(self.experiment_path, self.imputation_dict)
        logging.info('-'*60)
        return df
    
    def _get_null_columns(self, df):
        
        """Get null columns to fill them with a specific value. This returns all columns 
        in shortlisted features which are not in lab_ranges.keys(). lab_ranges are defined in config file.            
        """
        
        col_to_fill_na = [col for col in df.columns if col in list(set(self.shortlisted_features).difference(set(self.lab_ranges.keys())))]

        return col_to_fill_na

    def _check_update_duration_column(self, combined_df, duration_columns):
        
        """Checks that duration of diagnosis columns. \n
        If diagnosed = 1, increment duration by 1 only if duration < upper bound, else duration = 1 is returned. \n
        If diagnosed = 0, duration = 0 is returned.
        """
        duration_col_upper_bound = self.duration_col_upper_bound
        
        def check_duration_column(duration, diagnosis, upper_bound):
            """If diagnosis is 1 and duration is greater than upper_bound, set duration to 1. \n
            If diagnosis is 0, set duration to 0. \n
            Else, return the original duration +1.
            """
            duration_updated = np.where(
                (diagnosis == 1) & (duration > upper_bound), 1,
                np.where(diagnosis == 0, 0, duration+1)
            )
            return duration_updated
        
        for duration_column in duration_columns:
            diagnosis_code     = duration_column[-3:]
            diag_0_dur_not_0   = combined_df[(combined_df[diagnosis_code]==0)&(combined_df[duration_column]!=0)].shape[0]
            diag_1_dur_0       = combined_df[(combined_df[diagnosis_code]==1)&(combined_df[duration_column]==0)].shape[0]
            diag_1_dur_grtr_bnd= combined_df[(combined_df[diagnosis_code]==1)&(combined_df[duration_column]>duration_col_upper_bound)].shape[0]
            if diag_0_dur_not_0:
                logging.error(ValueError(f'There are {diag_0_dur_not_0} patients having {diagnosis_code} = 0 and {duration_column} != 0.'))
            if diag_1_dur_0:
                logging.info(f'There are {diag_1_dur_0} patients having {diagnosis_code} = 1 and {duration_column} = 0, where now {duration_column} = 1 is being replaced.')
            if diag_1_dur_grtr_bnd:
                logging.info(f'There are {diag_1_dur_grtr_bnd} patients having {diagnosis_code} = 1 and {duration_column} > {duration_col_upper_bound}, where now {duration_column} = 1 is being replaced.')
                
            combined_df[duration_column] = check_duration_column(combined_df[duration_column], combined_df[diagnosis_code], duration_col_upper_bound)
            
        return combined_df
    
    def _process_duration_columns(self, combined_df, shortlisted_features, print_int_stats=True):
        
        """This function converts the duration columns in combined df 
        into years instead of days. It also checks for 'duration_of_disease'
        column and does not alter that.
        """
        
        duration_columns = [x for x in shortlisted_features if 'duration' in x and x!='duration_of_disease']
        if duration_columns:
            logging.info(f'There are {len(duration_columns)}: {duration_columns} duration columns in your dataframe.')
            combined_df[duration_columns] = combined_df[duration_columns].apply(lambda x: x/365.25) # Convert days to years
            for column in duration_columns:
                combined_df = self._convert_to_int(combined_df, column, print_int_stats)
            combined_df = self._check_update_duration_column(combined_df, duration_columns)
                
        return combined_df
    
    def _group_features(self,combined_df):
        """
        For each group, this method checks if any of the specified features are present in the 
        DataFrame and groups them into a new column named after the group. The value of the group 
        column is determined by the following rules:
        - Set to 1 if any of the features in the group have a non-zero value.
        - Set to 0 if all of the features in the group are zero.
        - Set to None if none of the features are present or do not satisfy the above conditions.
        """
        logging.info('-------------------  FEATURE GROUPING  -------------------')
        for group_name, grouping_details in self.features_to_group.items():
            features         = grouping_details['features']
            grouped_features = [col for col in features if col in combined_df.columns]
            missing_features = set(features) - set(grouped_features)
            if missing_features:
                logging.warning(f"Features not found in DataFrame for group '{group_name}': {missing_features}")
    
            # Condition to check if any of the features are 1 or if all are 0, otherwise set to None
            condition = combined_df[grouped_features].any(axis=1).astype(int)
            condition[(combined_df[grouped_features] == 0).all(axis=1)] = 0
            condition[~combined_df[grouped_features].any(axis=1) & ~(combined_df[grouped_features] == 0).all(axis=1)] = None
    
            combined_df[group_name] = condition
            logging.info(f"Features grouped for '{group_name}': {grouped_features}")
        logging.info('-'*60)
        return combined_df
        
    def _process_combined_df(self, combined_df, print_int_stats,
                                                print_thresh_stats, 
                                                print_null_stats,
                                                print_map_stats):
        
        """This processes the combined df using various steps.
        """
         #------------------------------------------------------------------------------------------------------------------------#
        if self.grouping_required:
            combined_df = self._group_features(combined_df)
        #------------------------------------------------------------------------------------------------------------------------#
        combined_df = self._keep_non_null_within_threshold(combined_df, 'time_in_days', eval(self.config.get('FEATURE_RANGES', 'time_in_days_range')))
        logging.info('-'*60)
        logging.info("------------------  DROPPING ENTRIES WITH NULL VALUES  -------------------")
        for col in self.null_col_to_drop:
            if col in self.shortlisted_features:
                combined_df = self._drop_null_col(combined_df, column_to_drop=col)
        logging.info('-'*60)
        if 'age' in self.shortlisted_features:
            combined_df = self._convert_to_int(combined_df, 'age', print_int_stats)
        
        logging.info("------------------  DROPPING ENTRIES NOT WITHIN THRESHOLD  -------------------")
        for non_lab, range in self.non_lab_ranges.items():
            if non_lab not in self.shortlisted_features:
                continue
            combined_df = self._keep_non_null_within_threshold(combined_df, non_lab, range)
        logging.info('-'*60)
        # ----------------------------------------------------------------------------------- 
        logging.info("--------------------------  MAPPING  -----------------------")
        if isinstance(self.mapping, dict):
            
            for column, mapping in self.mapping.items(): 
                if column not in self.shortlisted_features:
                    continue
                combined_df = self._map_column(combined_df, column, mapping, print_map_stats)
        else:
            logging.error(ValueError('Since mapping in config is not dict, hence no mapping is being done.'))
        logging.info('-'*60)
        # -----------------------------------------------------------------------------------
        for lab, range in self.lab_ranges.items():
            if lab not in self.shortlisted_features:
                continue
            out_of_range_indices = combined_df[~((combined_df[lab]>=range[0])&(combined_df[lab]<=range[1])|combined_df[lab].isna())].index
            combined_df.loc[out_of_range_indices, lab] = None
            logging.info(f'There are {len(out_of_range_indices)} patients having {lab} value not in {range} range, which are being replaced with null.')
        # ------------------------------------------------------------------------------------
        self.utils.log_null_stats(combined_df, self.shortlisted_features)
        if self.imputation_required:
            combined_df = self._fill_null(combined_df, self._get_null_columns(combined_df))
        # ------------------------------------------------------------------------------------
        combined_df = self._process_duration_columns(combined_df, self.shortlisted_features, print_int_stats)
        combined_df = combined_df.copy() # To make it more memory efficient, since it is now a contigous block of memory
        
        return combined_df
    
    def _get_x_y_from_combined_df(self, df):
        """Returns x, y from combined df."""
        
        x_columns = list(df.columns)
        y_columns = ['is_diagnosed']
        
        for col in y_columns:
            x_columns.remove(col)
            
        X = df[x_columns]
        y = df[y_columns]
        
        return X, y
    
    def _get_x_with_selected_features(self, df):
        x = df[self.shortlisted_features].copy()
        return x
    
    def _get_y(self, df, is_survival):
        
        y_columns = ['is_diagnosed']
        if is_survival:
            y_columns.append('time_in_days')
        y = df[y_columns]
        
        if not is_survival:
            y = y['is_diagnosed']
            
        return y
    
    def _split_x_y(self, x, y):
        """Splits x, y into train, test (or val) splits."""
        
        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
        
        if self.val_required:
            x_train, x_val, y_train, y_val  = train_test_split(x_train_val, y_train_val, test_size=0.125, random_state=42,  shuffle=True)
            return x_train, x_test, x_val, y_train, y_test, y_val
        
        return x_train_val, x_test, y_train_val, y_test
    
    def _find_lab_mean_from_data(self, df, lab, clinical_range):
        """Find lab mean from data, if they are within range."""
        df = copy.deepcopy(df)
        df = df[df[lab].notna()]
        df = df[(df[lab]>=clinical_range[0]) & (df[lab]<=clinical_range[1])]
        return round(df[lab].mean(), 2)
                
    def _process_labs(self, train, test, val=None):
        """Imputing lab columns"""
        
        if self.imputation_required:
            for lab, range in self.lab_ranges.items():
                if lab not in train.columns:
                    continue
                mean = self._find_lab_mean_from_data(train, lab, range)
                self.imputation_dict['Imputation'][lab] = mean
                train[lab] = train[lab].fillna(mean)
                test [lab] = test [lab].fillna(mean)
                if val is not None:
                    val[lab] = val[lab].fillna(mean)
        
            self.utils.save_imputations(self.experiment_path, self.imputation_dict)
        # for lab, range in self.lab_ranges.items():
        #     if lab not in train.columns:
        #         continue
        #     train = self._keep_non_null_within_threshold(train, lab, range, print_thresh_stats=True)
        #     test  = self._keep_non_null_within_threshold(test, lab, range, print_thresh_stats=True)
        #     # train = train[((train[lab]>=range[0]) & (train[lab]<=range[1]))]
        #     # test  = test [((test [lab]>=range[0]) & (test[lab] <=range[1]))]
        #     if val is not None:
        #         val = self._keep_non_null_within_threshold(val, lab, range, print_thresh_stats=True)
        #         # val = val[((val[lab]>=range[0]) & (val[lab]<=range[1]))]
        
        if self.val_required:
            return train, test, val
        
        return train, test
    
    def _process_and_save_splits(self, train, test, val=None):
        
        
        if self.val_required:
            train, test, val = self._process_labs(train, test, val)
            self.utils.save_splits(self.experiment_path,  self.shortlisted_features, train, test, val)
            return train, test, val
        else:
            train, test = self._process_labs(train, test)
            self.utils.save_splits(self.experiment_path,  self.shortlisted_features, train, test)
            return train, test   
        
    def get_combined_data(self, diagnosed_filename = None,
                                normal_filename    = None,
                                filepath           = None,
                                balancing_required = None, 
                                verbose            = None):
        
        """
        Read dataframes and some preliminary preprocessing such as renaming, combining etc. before complete processing.
        """

        if diagnosed_filename is None:
            diagnosed_filename = self.config.get('FILE_SETTINGS', 'diagnosed_filename')
        if normal_filename is None:
            normal_filename = self.config.get('FILE_SETTINGS', 'normal_filename')
        if filepath is None:
            filepath = self.config.get('FILE_SETTINGS', 'dataset_filepath')
        if balancing_required is None:
            balancing_required = eval(self.config.get('BASE_SETTINGS', 'balancing_required'))
        if verbose is None:
            verbose = eval(self.config.get('BASE_SETTINGS', 'verbose'))


        experiment_path = self.experiment_path
        print_int_stats    = False
        print_thresh_stats = False
        print_null_stats   = False
        print_map_stats    = False
        
        if verbose:
            if verbose > 0:
                print_thresh_stats = True
            if verbose > 1:
                print_int_stats    = True
                print_map_stats    = True
            if verbose > 2:
                print_null_stats   = True
        
        utils = self.utils
        diagnosed = utils.load_data(True,  diagnosed_filename, filepath=filepath)
        normal    = utils.load_data(False, normal_filename,    filepath=filepath)
        logging.info('-'*60)
        logging.info('Printing shape after loading data from csv.')
        utils.print_shapes(diagnosed, normal)
        
        diagnosed['is_diagnosed'] = 1
        normal   ['is_diagnosed'] = 0
        
        diagnosed, normal = self.drop_columns(diagnosed, normal)
        # utils.print_shapes(diagnosed, normal)
        
        normal   = utils.rename_column(normal, utils.latest_encounter_clmn, utils.diagnosis_date_clmn)
        combined = self.concat_dataset(diagnosed, normal)
        logging.info(f'Combined shape: {combined.shape}')
        logging.info('-'*60)
        utils.check_class_imbalance(combined['is_diagnosed'], print_counts=True)
        utils.check_unique_patients(combined)
        
        combined = self.get_time_column(combined)
        
        logging.info('Processing dataset.')
        logging.info('-'*60)
        combined = self._process_combined_df(combined, 
                                            print_int_stats,
                                            print_thresh_stats,
                                            print_null_stats,
                                            print_map_stats)
        logging.info('-'*60)
        logging.info('Checking class imbalance after processing.')
        utils.check_class_imbalance(combined['is_diagnosed'], print_counts=True)
        logging.info('-'*60)
        if balancing_required:
            combined = utils.balance_dataset(combined)
            logging.info('-'*60)
        
        combined = utils.log_details(combined, self.shortlisted_features)
        utils.plot_time(combined, save_fig_path=experiment_path)
        
        return combined
    
    def make_splits(self, combined):
        
        """Make train, test, (and val) split from combined df.
        """
        
        X, y = self._get_x_y_from_combined_df(combined)
        if self.val_required:
            x_train, x_test, x_val, y_train, y_test, y_val = self._split_x_y(X, y)
        else:
            x_train, x_test, y_train, y_test = self._split_x_y(X, y)
            
        train = pd.concat([x_train, y_train], axis=1)
        test  = pd.concat([x_test,  y_test ], axis=1)
        if self.val_required:
            val   = pd.concat([x_val, y_val], axis=1)
        
        if self.val_required:
            train, test, val = self._process_and_save_splits(train, test, val)
            return train, test, val
        else:
            train, test      = self._process_and_save_splits(train, test)
            return train, test
    
    def get_splits(self, is_survival, train, test, val=None):
        
        """For given train, test, val splits, return corresponding x and y dataframes.
        """
    
        x_train = self._get_x_with_selected_features(train)
        y_train = self._get_y(train, is_survival)
        x_test  = self._get_x_with_selected_features(test)
        y_test  = self._get_y(test , is_survival)
        if val is not None:
            x_val = self._get_x_with_selected_features(val)
            y_val = self._get_y(val, is_survival)
            return x_train, x_test, x_val, y_train, y_test, y_val # Need to handle in training files if val is none
        return x_train, x_test, y_train, y_test 
    
    def get_data(self, is_survival):
        
        """Get train, test, val splits (with separate x, y dfs). If they do not exist; process the combined dataframe. 
        If they do exist, just read them.
        """
        
        splits_path = os.path.join(self.experiment_path, 'splits')  # correct this
        if os.path.exists(splits_path):
            train = pd.read_csv(f'{splits_path}/train.csv')
            test  = pd.read_csv(f'{splits_path}/test.csv')
            if self.val_required:
                val   = pd.read_csv(f'{splits_path}/val.csv')
                x_train, x_test, x_val, y_train, y_test, y_val = self.get_splits(is_survival=is_survival, train=train, test=test, val=val)
                return x_train, x_val, x_test, y_train, y_val, y_test
            x_train, x_test, y_train, y_test = self.get_splits(is_survival, train, test)
            return x_train, x_test, y_train, y_test
        
        combined = self.get_combined_data()
        if self.val_required:
            train, test, val = self.make_splits(combined)
            x_train, x_test, x_val, y_train, y_test, y_val = self.get_splits(is_survival=is_survival, train=train, test=test, val=val)
            return x_train, x_test, x_val, y_train, y_test, y_val
        else:
            train, test = self.make_splits(combined)
            x_train, x_test, y_train, y_test = self.get_splits(is_survival, train, test)
            return x_train, x_test, y_train, y_test
