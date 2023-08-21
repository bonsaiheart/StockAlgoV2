import logging

# Configure the root logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)

# Create a file handler that writes log messages to a file
file_handler = logging.FileHandler('UTILITIES/error.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s: %(message)s'))

# Add the file handler to the logger
logger.addHandler(file_handler)

# Add a test log entry
logger.info('This is a test log entry.')
