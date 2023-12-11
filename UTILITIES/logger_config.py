import datetime
import logging
import os
current_datetime=datetime.datetime.today()
formatted_date = current_datetime.strftime("%y%m%d")

print(formatted_date)
# Configthe root logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s: %(message)s')
log_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error_logs')
os.makedirs(log_directory, exist_ok=True)


# Create the log file path within the "error_logs" directory
log_file_path = os.path.join(log_directory, f'{formatted_date}_error.log')

logger = logging.getLogger(__name__)

# log messages to a file
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s: %(message)s'))

# Add the file handler to the logger
logger.addHandler(file_handler)

# Add a test log entry
# logger.info('This is a test log entry.')
