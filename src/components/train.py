import os 
import sys 
from dataclasses import dataclass 
import yaml

from catboost import CatBoostRegressor 
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

yaml_file = './src/models.yaml'
def load_models(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    model_params = data['models']
    
    models = {}
    for model_name, model_info in model_params.items():
        model_class = globals().get(model_info['model'])
        if model_class:
            models[model_name] = {
                'model': model_class(),
                'params': model_info['params']
            }
    return models 

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'best_model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self,train_arr, test_arr):
        try:
            logging.info("Splitting Training and Test Input Data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = load_models(yaml_file)
            
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,models=models)
            
            # to get the best model score from the dict 
            best_model_score = max(sorted(model_report.values()))
            
            # to get the best model name from dict 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # Get the model instance for the best model
            best_model = models[best_model_name]['model']
            
            if best_model_score < 0.6:
                raise CustomException("No best model found!")
            
            # Save the best model to file
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            
            logging.info("Best model saved as .pkl")
            
            # Predict using the best model
            predicted = best_model.predict(X_test)
            r2score = r2_score(y_test, predicted)
            return r2score, best_model
            
        except Exception as e:
            raise CustomException(e, sys)