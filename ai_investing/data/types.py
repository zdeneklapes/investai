# -*- coding: utf-8 -*-


from typing import Literal
import enum


TimeInterval = Literal["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]


class FileType(enum.Enum):
    pass


#     @classmethod
#     def list(cls):
#         return [e.value for e in cls]
#
#     JSON = "json"
#     CSV = "csv"
#


class DataType(enum.Enum):
    pass


#     TRAIN = "train"
#     TEST = "test"
