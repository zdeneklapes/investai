# -*- coding: utf-8 -*-
from enum import IntEnum, auto


class BaseExitCode(IntEnum):
    """Exit codes for training process are in range 0000-1000."""

    OK: int = auto()
    BAD_ARGS: int = auto()
    BAD_PARAMS: int = auto()
    BAD_DATA: int = auto()
    BAD_MODEL: int = auto()
    BAD_TRAIN: int = auto()
    BAD_EVAL: int = auto()
    BAD_PREDICT: int = auto()
    BAD_SAVE: int = auto()


if __name__ == "__main__" and "__file__" in globals():
    print(BaseExitCode)
