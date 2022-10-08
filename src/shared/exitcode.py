import sys
from os import path
from typing import Dict


class ExitCode:
    OK = 0
    BAD_ARGS = 10
    BAD_PARAMS = 20

    @staticmethod
    def check_paths(*, params: Dict[str, str]):
        for key, val in params.items():
            if key.find("dir") != -1:
                if not path.isdir(str(val)):
                    # TODO: LOGGER_STREAM.error(f'Bad directory|{key}: {val}')
                    print(f"Bad directory|{key}: {val}")  # TODO: Remove this
                    sys.exit(ExitCode.BAD_PARAMS)
