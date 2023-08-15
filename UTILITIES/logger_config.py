import logging

logging.basicConfig(filename="errorlog/error.log", level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
