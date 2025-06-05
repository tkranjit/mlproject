import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_models

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor




from src.utils import save_object,evaluate_models

@dataclass  
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """        Initiates the model training process.
        Args:
            train_array (numpy.ndarray): Training data array.
            test_array (numpy.ndarray): Testing data array.
            preprocessor_object: Preprocessing object (not used in this method).    
        Returns:
            tuple: Best model name and its R2 score.            
        """      
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
                "XGBRegressor": XGBRegressor()
            }

            model_report = {}

            model_report = evaluate_models(
                X_train, y_train, X_test, y_test, models
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)  

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")
            best_model= models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found with R2 score greater than 0.6", sys
                )
            logging.info("Saving the best model")   
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model training completed successfully")
            predicted_best_model = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted_best_model)
            logging.info(f"R2 score of the best model: {r2_square}")
            logging.info(f"Best model name: {best_model_name}")
            logging.info(f"Best model score: {best_model_score}")
            return best_model_name, r2_square

        except Exception as e:
            raise CustomException(e, sys)

