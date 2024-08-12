'''# in utils module we only stores the common functionalitis .
so we can access it multiple time from this modeule to another module'''

import os
import sys
import dill # this library is used to create the pickle file

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from src.exception import CustomeException


def save_objects(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path ,exist_ok=True)

        file_obj = open(file_path ,"wb")
        dill.dump(obj , file_obj)

    except Exception as e:
        raise CustomeException(e  , sys)



# creating function for model evaluation
def evaluate_model(x_train , y_train , x_test , y_test , models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(x_train , y_train )
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            model_r2_score_training = r2_score(y_train , y_train_pred)
            model_r2_score_testing =r2_score(y_test , y_test_pred)

            report[list(models.keys())[i]] = model_r2_score_testing

        return report
    except Exception as e :
        raise CustomeException(e , sys)
