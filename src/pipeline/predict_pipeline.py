import sys
import os
import pandas as pd
from src.logger import logging  
from src.exception import CustomException
from src.utils import load_object, save_object, evaluate_models
from dataclasses import dataclass





class PredictPipeline:
    def __init__(self):
        pass
    def predict_predictions(self, features):
        try:
            
            logging.info("Starting prediction pipeline.")
            print("Starting prediction pipeline.")
            # Load the preprocessor object
            preprocessor = load_object("src/components/artifacts/preprocessor.pkl")
            
            logging.info(f"Preprocessor loaded successfully from artifacts/preprocessor.pkl")
            print(f"Preprocessor loaded successfully from artifacts/preprocessor.pkl")
            # Transform the input features
            data_scaled = preprocessor.transform(features)
            
            # Load the trained model
            model = load_object("src/components/artifacts/model.pkl")
            logging.info(f"Model loaded successfully from artifacts/model.pkl")
            print(f"Model loaded successfully from artifacts/model.pkl")
            
            # Make predictions
            predictions = model.predict(data_scaled)
            logging.info(f"Predictions made successfully: {predictions}")
            print(f"Predictions made successfully: {predictions}")
            
            return predictions
        
        except CustomException as e:
            logging.error(f"Error in PredictPipeline: {e}")
            raise CustomException(e, sys) 

class customData:
    def __init__(self,gender: str, race_ethnicity: str,parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 writing_score: int,
                 reading_score: int):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.writing_score=writing_score
        self.reading_score=reading_score
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "writing_score": [self.writing_score],
                "reading_score": [self.reading_score]
            }
            logging.info(f"Custom data input dictionary: {custom_data_input_dict}")
            return pd.DataFrame(custom_data_input_dict)
        
        except CustomException as e:
            logging.error(f"Error in get_data_as_dataframe: {e}")
            raise CustomException(e, sys)

    