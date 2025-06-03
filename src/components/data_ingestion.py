from dataclasses import dataclass
import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd

import sklearn.model_selection as model_selection
@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')      
    raw_data_path = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("../../notebook\data\stud.csv")
            logging.info("Read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            logging.info("Created directory for artifacts if it did not exist")
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts directory")
            train_set,test_set=model_selection.train_test_split(df,test_size=0.2,random_state=1102)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info("Train set saved to artifacts directory")  
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Test set saved to artifacts directory")
            logging.info("Data ingestion completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            logging.error(f"An error occurred during data ingestion: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
    print("Data ingestion completed successfully")
        

