# -*- coding: utf-8 -*-
from enum import IntEnum, auto


class ExitCode(IntEnum):
    """Exit codes for training process are in range 0000-1000."""

    OK: int = auto()
    ERROR: int = auto()


if __name__ == "__main__" and "__file__" in globals():
    print(ExitCode)
