# in this file we just getting the data like from api, from databases......
import os
import sys
from src.exception import CustomeException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass # it use to names the classes

@dataclass  # this is decorator which help us without intializing the function
# if u want to give input then u have to use the config word
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts' , 'train.csv') # all inputs output store inside artifacts folder
    test_data_path:str=os.path.join('artifacts' , 'test.csv')
    raw_data_path:str=os.path.join('artifacts' , "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method or component")
        try:
            data = pd.read_csv('notebook/dataset/StudentsPerformance.csv')
            logging.info('Read the dataset as DataFrame')

            # creating the artifact folder and store logs inside it
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path) , exist_ok=True)

            #saving the data inside the artifact directory
            data.to_csv(self.ingestion_config.raw_data_path , index=False , header=True)

            logging.info("train test split initiated")

            #splitting the data
            train_data , test_data  = train_test_split(data , test_size=0.2 , random_state=43)

            # saving train data to train.csv and test data to test.csv
            train_data.to_csv(self.ingestion_config.train_data_path , index=False , header=True)
            test_data.to_csv(self.ingestion_config.test_data_path , index=False , header=True)

            logging.info("ingestion of data inside the csv files")

            return (self.ingestion_config.train_data_path ,
                    self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomeException(e ,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()