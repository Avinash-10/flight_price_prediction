import pandas as pd
import numpy as np
import os
import sys
import functools

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object
import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTransformationConfig()

    def get_datatransform_obj(self):
        try:
            numerical_column = ['duration', 'days_left']
            categorical_column = ['airline', 'source_city', 'departure_time', 'stops',
                                  'arrival_time', 'destination_city', 'class']

            numerical_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaling',StandardScaler(with_mean=False))
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('labelencoder',OneHotEncoder()),
                ('scaling',StandardScaler(with_mean=False))
                ]
            )

            logging.info('Numerical columns standard scaling completed')
        
            logging.info('Catergorical column encoding completed')


            preprocessor = ColumnTransformer(
                [
                ('numerical columns',numerical_pipeline,numerical_column),
                ('catergorical columns',categorical_pipeline,categorical_column)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiating_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info('Train Test data has been read')

            logging.info('Obtaining preprocessng data')

            preprocessor_obj = self.get_datatransform_obj()

            target_column = 'price'
            drop_col = [target_column,'Unnamed: 0','flight']
            # numerical_column = ['duration', 'days_left','Unnamed: 0']

            input_feature_train_data = train_data.drop(columns=drop_col, axis=1)
            target_feature_train_data = train_data[target_column]

            input_feature_test_data = test_data.drop(columns=drop_col, axis=1)
            target_feature_test_data = test_data[target_column]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_data_arr = preprocessor_obj.fit_transform(input_feature_train_data)
            input_feature_test_data_arr = preprocessor_obj.transform(input_feature_test_data)

            # np.concatenate((a,b[:,None]),axis=1)

            train_arr = np.hstack((input_feature_train_data_arr, np.array(target_feature_train_data)))
            test_arr = np.hstack((input_feature_test_data_arr,np.array(target_feature_test_data)))
            

            logging.info("Saved preprocessing object.")
            
            save_object(
                file_path = self.data_transform_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transform_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)



