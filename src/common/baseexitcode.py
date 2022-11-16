# -*- coding: utf-8 -*-


class BaseExitCode:
    """Exit codes for training process are in range 0000-1000."""

    OK = 0
    BAD_ARGS = 1
    BAD_PARAMS = 2


class TrainExitCode(BaseExitCode):
    """Exit codes for training process are in range 1000-1999."""

    OK = 1001
    BAD_ARGS = 1002
    BAD_PARAMS = 1003
    BAD_DATA = 1004
    BAD_MODEL = 1005
    BAD_TRAIN = 1006
    BAD_EVAL = 1007
    BAD_PREDICT = 1008
    BAD_SAVE = 1009


class TestExitCode(BaseExitCode):
    """Exit codes for training process are in range 2000-2999."""

    OK = 2001
    BAD_ARGS = 2002
    BAD_PARAMS = 2003
    BAD_DATA = 2004
    BAD_MODEL = 2005
    BAD_TRAIN = 2006
    BAD_EVAL = 2007
    BAD_PREDICT = 2008
    BAD_SAVE = 2009
