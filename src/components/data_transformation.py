# in this file we do the all transformation of like encoding ,  dtypes and different transformation technoques we problably going to use


'''
1) in this file we do the all transformation of like encoding ,  dtypes and different transformation
technoques we problably going to use
'''

import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomeException
from src.logger import logging
from src.utils import save_objects
import os


@dataclass
class  DataTransformationConfig:
    processed_obj_file_path = os.path.join("artifacts" , "processor.pkl")


class DataTransformation:
    def __init__(self):
        self.DataTransformationConfig = DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        this fucntion help us to transform the data. for both numerical as well as categorical data
        '''
        logging.info('started')
        try:
            numerical_features = ['reading score', 'writing score']
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']


            # we have to create two pipe lines for numerical data and for categorical data
            num_pipeline = Pipeline(
                steps=[
                    # we using imputer for handling the missing values
                    ("imputer" , SimpleImputer(strategy="median",missing_values=np.nan)) ,
                    ("standardscaler" , StandardScaler())
                ]
            )
            logging.info("numerical data scaled successfully ")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(sparse_output=False)),  # Ensuring dense output
                    ("scaler", StandardScaler(with_mean=False))  # or MaxAbsScaler if you want to keep sparsity
                ]
            )
            logging.info("categorical data transformed successfully ")


            # now combining both the pipelines and fitting data inside it
            preprocessor = ColumnTransformer([
                ("num_features" , num_pipeline , numerical_features) ,
                ("cat_features" , cat_pipeline , categorical_features)
            ])

            return preprocessor
        except Exception as e:
            raise CustomeException(e , sys)


    # creating the function for initializing the data transformation
    def initiate_data_transformation(self , train_path , test_path):

        logging.info('initaited ')
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read  training and testing data completed")

            logging.info("obtaining preprocessing object")

            # creating the object of class get_data_transformer_obj
            preprocess_obj = self.get_data_transformer_obj()
            target_column = 'math score'
            numerical_columns = ['reading score', 'writing score']

            input_feature_train = train_df.drop(columns=[target_column] , axis=1)
            input_feature_test = test_df.drop(columns=[target_column] , axis=1)

            logging.info('applying preprocessing object on train and test dataframe')

            # taking target features label
            target_train_df = train_df[target_column]
            target_test_df = test_df[target_column]
            input_data_train_arr = preprocess_obj.fit_transform(input_feature_train)
            input_data_test_arr = preprocess_obj.transform(input_feature_test)

            train_arr = np.c_[input_data_train_arr , np.array(target_train_df)]
            test_arr = np.c_[input_data_test_arr , np.array(target_test_df)]

            logging.info("saving the processed data.............👌👍")


            # this function is present in  utils module
            save_objects(
                file_path = self.DataTransformationConfig.processed_obj_file_path,
                obj = preprocess_obj

            )

            return (train_arr , test_arr ,
                    self.DataTransformationConfig.processed_obj_file_path )

        except Exception as e:
            raise CustomeException(e ,sys)

