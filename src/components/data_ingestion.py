import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    clean_data_path: str = os.path.join('artifacts','clean_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiating_data_ingestion(self):
        logging.info('Entered the Data Ingestion method')

        try:
            dataframe = pd.read_csv('Data\Clean_Dataset.csv')
            logging.info('Data is exported and read in a Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            dataframe.to_csv(self.ingestion_config.clean_data_path, index = False, header = True)

            logging.info('Train Test split initiated')
            train_set, test_set = train_test_split(dataframe, test_size = 0.2, random_state=0)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('Ingestion of Data is Completed')

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiating_data_ingestion()