import os
import sys
from src.exception import CustomException
from src.logger import logging

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.utils import save_object
from src.utils import evaluate_models
from xgboost import XGBRegressor

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        self.model=None
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),    
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "AdaBoost Regressor":AdaBoostRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor()
            }

            params={
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'max_depth': [3, 4, 5, 6, 8, 10, 12, 15, 18],
                    'n_estimators': [8,16,32,64,128,256]    
                },
                "Linear Regression":{},
                "AdaBoost Regressor":{
                    'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighborsRegressor":{
                    'n_neighbors': [2,3,4,5,6,7,8,9,10],
                    'weights': ['uniform','distance'],
                    'algorithm': ['auto','ball_tree','kd_tree','brute']
                }            
            }
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model  
            )

            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test,predicted)    

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)    