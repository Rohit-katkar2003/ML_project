'''# in utils module we only stores the common things.
so we can access it multiple time from this modeule to another module'''

import os
import sys
import dill # this library is used to create the pickle file

import pandas as pd
import numpy as np
from src.exception import CustomeException


def save_objects(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path ,exist_ok=True)

        file_obj = open(file_path ,"wb")
        dill.dump(obj , file_obj)

    except Exception as e:
        raise CustomeException(e  , sys)