import logging
import os
from datetime import datetime

#### monitor all process in log 

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" 
log_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(log_path,exist_ok=True)

LOG_FILE_Path = os.path.join(log_path,LOG_FILE)

logging.basicConfig(
        filename= LOG_FILE_Path,
        format= "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level= logging.INFO,

)

# if __name__=="__main__":
#     logging.info("Logging has started")   #exception occur in terminal