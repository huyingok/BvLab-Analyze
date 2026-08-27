# -*- coding: utf-8 -*-
import logging.config


class setLoggerConfig(object):
    def __init__(self):
        self.logger = None

    def get_logger_object(self, logging_path):
        root_level_str = 'ERROR'
        loggers_level_str = 'INFO'
        _config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': ' %(asctime)s - %(name)s[line:%(lineno)d] - %(levelname)s - %(message)s',
                    'datefmt': "%Y-%m-%d %H:%M:%S"
                }
            },
            'handlers': {
                'rotatingHandler': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': logging_path,
                    'maxBytes': 1024 * 1024 * 1024,
                    'backupCount': 3,
                    'formatter': 'default',
                    'encoding': 'utf-8',
                },
                'timedHandler': {
                    'class': 'logging.handlers.TimedRotatingFileHandler',
                    'filename': logging_path,
                    'when': 'midnight',
                    'interval': 1,
                    'backupCount': 7,
                    'formatter': 'default',
                    'encoding': 'utf-8',
                },
            },
            'loggers': {
                'celery': {
                    'handlers': ['timedHandler'],
                    'level': loggers_level_str,
                    'propagate': False
                },
            },
            'root': {
                'handlers': ['rotatingHandler'],
                'level': root_level_str,
                'propagate': False
            },
        }

        logging.config.dictConfig(_config)
        logger = logging.getLogger('appLog')

        return logger

