from os import path
import sys
from typing import Dict
from shared.logging_setup import LOGGER_STREAM


class ExitCode:
    OK = 0
    BAD_ARGS = 10
    BAD_PARAMS = 20

    @staticmethod
    def check_paths(*, params: Dict[str, str]):
        for key, val in params.items():
            if key.find('dir') != -1:
                if not path.isdir(str(val)):
                    LOGGER_STREAM.error(f'Bad directory|{key}: {val}')
                    sys.exit(ExitCode.BAD_PARAMS)
