# it see all the logs
import logging
import os
from datetime import datetime

log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd() , 'logs' , log_file)
os.makedirs(log_path , exist_ok=True)


Log_file_Path =os.path.join(log_path , log_file)

logging.basicConfig(
    filename=Log_file_Path ,
    format=f"[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO ,
)

if __name__ == "__main__":
    logging.info("Logging has started")