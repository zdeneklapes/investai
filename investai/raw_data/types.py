# -*- coding: utf-8 -*-


import enum
from typing import Literal

TimeIntervalType = Literal["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]


class FileType(enum.Enum):
    pass


class DataType(enum.Enum):
    pass
