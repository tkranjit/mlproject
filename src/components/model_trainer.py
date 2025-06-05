import os
import sys

from src.logger import logging
from src.exception import CustomException

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
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=0),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report = {}

            model_report = evaluate_models(
                X_train, y_train, X_test, y_test, models,params
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
            print("model trained and saved successfully")
            print(f"R2 score of the best model: {r2_square}")
            print(f"Best model name: {best_model_name}")
            return best_model_name, r2_square

        except Exception as e:
            raise CustomException(e, sys)

