# this file contains error handling and exception handling
import sys
import logging
def error_message_details(error , error_detail:sys):

    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"error occured in script name [{file_name}] in line number [{exc_tb.tb_lineno}] error meassage is [{str(error)}]"

    return  error_message

class CustomeException(Exception):
    def __init__(self,error_message , error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message ,error_detail=error_details)

    def __str__(self):
        return self.error_message

if __name__=="__main__":
    try :
        a = 1/0
    except Exception as e:
        logging.info("divide by zero")
        raise CustomeException(e , sys)