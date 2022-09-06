# from resources.logger.src.logging_helper import *
# logger_stdout = get_logger_stdout()

import os
from os import path
import sys
import logging

import colorama
from colorama import Fore

from settings import ROOT_DIR, SRC_DIR, AI_ROOT_DIR

##########################################################
if 0:
    logging.debug('Debug message')
    logging.info('Info message')
    logging.warning('Warning message')
    logging.error('Error message')
    logging.critical('Critical message')
elif 0:
    logging.basicConfig(filename='file.log',
                        level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    logging.debug('Debug message')
    logging.info('Info message')
    logging.warning('Warning message')
    logging.error('Error message')
    logging.critical('Critical message')
elif 0:
    # Create `parent.child` logger
    LOGGER_STDOUT = logging.getLogger("parent.child")

    # Emit a log message of level INFO, by default this is not print to the screen
    LOGGER_STDOUT.info("this is info level")

    # Create `parent` logger
    parentlogger = logging.getLogger("parent")

    # Set parent's level to INFO and assign a new handler
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s = %(message)s"))
    parentlogger.setLevel(logging.INFO)
    parentlogger.addHandler(handler)

    # Let child logger emit a log message again
    LOGGER_STDOUT.info("this is info level again")
elif 0:
    # Set up root logger, and add a file handler to root logger
    logging.basicConfig(filename='file.log',
                        level=logging.WARNING,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    # Create logger, set level, and add stream handler
    parent_logger = logging.getLogger("parent")
    parent_logger.setLevel(logging.INFO)
    parent_shandler = logging.StreamHandler()
    parent_logger.addHandler(parent_shandler)

    # Log message of severity INFO or above will be handled
    parent_logger.debug('Debug message')
    parent_logger.info('Info message')
    parent_logger.warning('Warning message')
    parent_logger.error('Error message')
    parent_logger.critical('Critical message')
elif 1:
    pass


# source: https://machinelearningmastery.com/logging-in-python/
def loggingdecorator(name):
    logger = logging.getLogger(name)

    def _decor(fn):
        function_name = fn.__name__

        def _fn(*args, **kwargs):
            ret = fn(*args, **kwargs)
            argstr = [str(x) for x in args]
            argstr += [key + "=" + str(val) for key, val in kwargs.items()]
            logger.debug("%s(%s) -> %s", function_name, ", ".join(argstr), ret)
            return ret

        return _fn

    return _decor


# Prepare the colored formatter
colorama.init(autoreset=True)
colors = {"DEBUG": Fore.LIGHTBLACK_EX,
          "INFO": Fore.CYAN,
          "WARNING": Fore.YELLOW,
          "ERROR": Fore.RED,
          "CRITICAL": Fore.MAGENTA}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.levelname in colors:
            msg = colors[record.levelname] + msg + Fore.RESET
        return msg


# Create logger and assign handler
LOGGER_STDOUT = logging.getLogger("main1")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter("%(asctime)s|%(filename)s:%(lineno)s|%(levelname)s|%(name)s|%(message)s"))
LOGGER_STDOUT.addHandler(handler)
LOGGER_STDOUT.setLevel(logging.DEBUG)

# Create logger and assign handler
LOGGER_STDERR = logging.getLogger("main1")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(ColoredFormatter("%(asctime)s|%(filename)s:%(lineno)s|%(levelname)s|%(name)s|%(message)s"))
LOGGER_STDERR.addHandler(handler)
LOGGER_STDERR.setLevel(logging.DEBUG)


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


h1 = logging.StreamHandler(sys.stdout)
f1 = SingleLevelFilter(logging.INFO, False)
h1.addFilter(f1)
rootLogger = logging.getLogger()
rootLogger.addHandler(h1)
h2 = logging.StreamHandler(sys.stderr)
f2 = SingleLevelFilter(logging.INFO, True)
h2.addFilter(f2)
rootLogger.addHandler(h2)

# info: stdout, others: stderr
LOGGER_STREAM = logging.getLogger("my.logger")
LOGGER_STREAM.setLevel(logging.DEBUG)

# LOGGER_STREAM.debug("A DEBUG message")
# LOGGER_STREAM.info("An INFO message")
# LOGGER_STREAM.warning("A WARNING message")
# LOGGER_STREAM.error("An ERROR message")
# LOGGER_STREAM.critical("A CRITICAL message")

file = path.join(AI_ROOT_DIR, 'logs', 'logs.log')

if not path.isfile(file):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    open(file, 'x').close()

# Create logger and assign handler
LOGGER_FILE = logging.getLogger("main2")
handler = logging.FileHandler(path.join(AI_ROOT_DIR, 'logs', 'logs.log'))
handler.setFormatter(logging.Formatter("%(asctime)s|%(filename)s:%(lineno)s|%(levelname)s|%(name)s| %(message)s"))
LOGGER_FILE.addHandler(handler)
LOGGER_FILE.setLevel(logging.DEBUG)

# examples
# LOGGER_FILE.debug('foo')
# LOGGER_FILE.info('foo')
# LOGGER_FILE.warning('foo')
# LOGGER_FILE.error('foo')
# LOGGER_FILE.critical('foo')
