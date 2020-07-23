import logging
import os


def get_logger(name):
    log_format = '%(asctime)s  %(name)8s  %(levelname)5s  %(message)s'
    log_file = 'dev.log'
    log_path = os.path.join("..", "logs", log_file)
    logging.basicConfig(level=logging.DEBUG,
                        format=log_format,
                        filename=log_path,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(name).addHandler(console)
    return logging.getLogger(name)
