import logging
import os

# Configthe root logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s: %(message)s')
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error.log')

logger = logging.getLogger(__name__)

# log messages to a file
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s: %(message)s'))

# Add the file handler to the logger
logger.addHandler(file_handler)

# Add a test log entry
# logger.info('This is a test log entry.')
