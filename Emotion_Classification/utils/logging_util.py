import os
import datetime
import logging


def get_logging():
    # 日志模块
    TODAY = datetime.date.today()
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y/%m/%d %H:%M:%S %p"
    LOG_DIR = f'output/log/'
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    logging.basicConfig(
                        filename=f"./output/log/{TODAY}.log", 
                        level=logging.DEBUG, 
                        format=LOG_FORMAT, 
                        datefmt=DATE_FORMAT
                    )
    
    return logging