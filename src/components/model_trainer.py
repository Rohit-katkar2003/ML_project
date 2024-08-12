# in this file we basically training our model and evaluating it's performance


# importing the dependencies
import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor ,  AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score


from src.exception import  CustomeException
from src.logger import logging
from src.utils import save_objects
from src.utils import evaluate_model


# creating my model training pickle file
@dataclass # decorator use for  allows a user to add new functionality to an existing object without modifying its structure
class ModelTrainingConfig:
    model_training_file = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
    def initiate_model_trainer(self , train_arr , test_arr  ):
        try:
            logging.info("initializing the model training")
            # now we simplily splitting data into training and testing
            x_train ,x_test ,y_train , y_test =train_arr[: , :-1] , test_arr[: ,:-1] , train_arr[: , -1] , test_arr[: ,-1]

            # creating the dictionary of all model which we are going to use
            models = {"linearregression":LinearRegression() ,
                      "randomForest" : RandomForestRegressor(n_estimators=150) ,
                      "decisionTree": DecisionTreeRegressor() ,
                      "kNearestNeighour":KNeighborsRegressor(n_neighbors=5) ,
                      "GradientBoosting" : GradientBoostingRegressor() ,
                      "Adaboost":AdaBoostRegressor() ,
                      "xgboost" :XGBRegressor() ,
                      'catboost':CatBoostRegressor(verbose=False)}
            model_report:dict=evaluate_model(x_train=x_train , y_train=y_train , x_test=x_test
                                             , y_test=y_test , models=models)

            ## to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## best model  name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise CustomeException("no best model found")

            logging.info("we got the best model ")
            save_objects(
                file_path=self.model_trainer_config.model_training_file ,
                obj= best_model
            )

            predicted = best_model.predict(x_test)

            r2_squared = r2_score(y_test , predicted)
            return r2_squared

        except Exception as e:
            raise CustomeException(sys ,e)



