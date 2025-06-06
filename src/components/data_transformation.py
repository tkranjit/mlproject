
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os
import sys

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features =[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            logging.info("Creating numerical and categorical transformers") 
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])  
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])  
            logging.info("Creating numerical and categorical transformers standard scaling completed")
            logging.info("catogorical features: %s", categorical_features)
            logging.info("numerical features: %s", numerical_features)
            preporcessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
            return preporcessor
        except Exception as e:
            logging.error(f"An error occurred while creating the data transformer object: {e}")
            raise CustomException(e, sys)   
        
    def initiate_data_transformation(self, train_path, test_path):
            try:
                train_df = pd.read_csv(train_path)
                
                test_df = pd.read_csv(test_path)
                logging.info("Read train and test dataframes successfully")
                
                preprocessing_obj = self.get_data_transformer_object()
                target_column_name = "math_score"
                
                input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
                target_feature_train_df = train_df[target_column_name]
                
                input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
                target_feature_test_df = test_df[target_column_name]
                
                logging.info("Applying preprocessing object on training and testing dataframes")
                
                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
                
                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
                
                logging.info("Data transformation completed successfully")
                
                save_object(
                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj)

            
                return train_arr, test_arr, preprocessing_obj
            except Exception as e:
                logging.error(f"An error occurred during data transformation: {e}")
                raise CustomException(e, sys)   
            
