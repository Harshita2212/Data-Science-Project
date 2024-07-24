import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from catboost import CatBoostRegressor # type: ignore
from sklearn.ensemble import (         # type: ignore
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
) 
from sklearn.linear_model import LinearRegression     # type: ignore
from sklearn.metrics import r2_score                  # type: ignore
from sklearn.neighbors import KNeighborsRegressor     # type: ignore
from sklearn.tree import DecisionTreeRegressor        # type: ignore
from xgboost import XGBRegressor                      # type: ignore

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split Training and test input data")
            X_train, Y_train, X_test, Y_test = (
                train_array[:,:-1], #except last column
                train_array[:,-1],  #Only Last column
                test_array[:,:-1],  #except last column
                test_array[:,-1]    #Only Last column
            )

            #create dectionary
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            #To add Hyperparamerter training use params as dictionary
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8,16,32,64,128,256],
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [5,7,9,11],
                },
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor": {
                    'depth': [6,8,10],
                    'learning_rate': [.1, .01, .05, .001],
                    'iterations': [30,50,100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report:dict = evaluate_model(
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test, 
                models=models,
                params=params
            )

            #to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ##to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best found model on both training and testing dataset")

            #we can load preprocessor.pkl file in preprocessor_obj but here we don't need 
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            ) 

            predicted = best_model.predict(X_test)
            r2_square = r2_score(Y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
