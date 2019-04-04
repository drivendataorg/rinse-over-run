import json
import logging
import sys
from logging import handlers

loggers = {}


class Logger:
    """
    Training process logger

    Note:
        Used by BaseTrainer to save training history.
    """

    def __init__(self, name=''):
        global loggers
        if loggers.get(name):
            self.logger = loggers.get(name)
        else:
            # logging.basicConfig(filename="./file.log.h", level=logging.INFO)
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)
            self.logger = logging.getLogger(name)
            self.entries = {}
            self.logger.propagate = False
            handler = logging.handlers.TimedRotatingFileHandler("./file.log.h")
            handler.setLevel(logging.INFO)
            self.logger.addHandler(handler)
            loggers[name] = self.logger

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)
