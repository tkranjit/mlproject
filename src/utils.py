import os
import sys

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging  
import pickle
import numpy as np  
import pandas as pd

def load_object(file_path):
    """
    Load an object from a file using pickle.
    
    Parameters:
    - file_path: str, path to the file from which the object will be loaded.
    
    Returns:
    - obj: object, the loaded object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        raise CustomException(e, sys)

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    
    Parameters:
    - file_path: str, path where the object will be saved.
    - obj: object, the object to be saved.
    
    Returns:
    None
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}")
        raise CustomException(e, sys)  
    
def evaluate_models(X_train, y_train, X_test, y_test, models,params=None):
    """
    Evaluate multiple regression models and return their performance metrics.
    
    Parameters:
    - X_train: np.ndarray, training feature set.
    - y_train: np.ndarray, training target variable.
    - X_test: np.ndarray, testing feature set.
    - y_test: np.ndarray, testing target variable.
    - models: dict, dictionary of model names and their instances.
    
    Returns:
    - model_report: dict, model names as keys and their R2 scores as values.
    """
    model_report = {}
    
    for i in range(len(list(models))):
        model = list(models.values())[i]
        para=params[list(models.keys())[i]] 

        gs=GridSearchCV(model,para,cv=3)
        gs.fit(X_train, y_train)

        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_square = r2_score(y_test, y_pred)
        model_report[list(models.keys())[i]] = r2_square    
        logging.info(f"{list(models.keys())[i]}: R2 score = {r2_square}")
    logging.info("Model evaluation completed successfully")
    return model_report