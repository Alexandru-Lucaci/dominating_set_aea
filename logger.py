import logging
import time

# Default logging level
loggingLevel = logging.INFO


class Logger:
    def __init__(self, log_file_name):
        self.log_file_name = log_file_name
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.handler = logging.FileHandler(log_file_name)
        self.handler.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

    def log(self, message: str, level=logging.INFO):
        if level == logging.INFO:
            self.logger.info(message)
            print(f"[INFO] [{time.strftime('%d-%m %H:%M:%S', time.localtime())}] : {message}")
        elif level == logging.ERROR:
            self.logger.error(message)
        elif level == logging.WARNING and loggingLevel == logging.WARNING:
            print(f"[WARNING] [{time.strftime('%d-%m %H:%M:%S', time.localtime())}] : {message}")
        # elif level == logging.WARNING:
        #     self.logger.warning(message)

    def close(self):
        self.handler.close()