# logging_config.py
import logging

def configure_logger(name):
    # Create a logger with the provided name
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set level of logger

    # Create a file handler
    handler = logging.FileHandler('logfile.log', 'w')
    handler.setLevel(logging.INFO)  # Set level of handler

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)

    return logger
