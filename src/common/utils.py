# -*- coding: utf-8 -*-
import sys
from os import path
from typing import (
    Dict,
)

from common.baseexitcode import BaseExitCode


class Util:
    @staticmethod
    def check_paths(*, params: Dict[str, str]):
        for key, val in params.items():
            if key.find("dir") != -1 and not path.isdir(str(val)):
                # TODO: LOGGER_STREAM.error(f'Bad directory|{key}: {val}')
                print(f"Bad directory|{key}: {val}")  # TODO: Remove this
                sys.exit(BaseExitCode.BAD_PARAMS)


def now_time(format: str = "%Y-%m-%dT%H-%M-%S") -> str:
    import datetime

    return datetime.datetime.now().strftime(format)
