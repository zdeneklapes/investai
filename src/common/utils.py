# -*- coding: utf-8 -*-
##
from os import path
import sys
from typing import Dict
import cProfile
import pstats

##
from common.baseexitcode import BaseExitCode


class Util:
    @staticmethod
    def check_paths(*, params: Dict[str, str]):
        for key, val in params.items():
            if key.find("dir") != -1 and not path.isdir(str(val)):
                # TODO: LOGGER_STREAM.error(f'Bad directory|{key}: {val}')
                print(f"Bad directory|{key}: {val}")  # TODO: Remove this
                sys.exit(BaseExitCode.BAD_PARAMS)


def now_time(_format: str = "%Y-%m-%dT%H-%M-%S") -> str:
    import datetime

    return datetime.datetime.now().strftime(_format)


def line_profiler_stats(func):
    def wrapper(*args, **kwargs):
        import line_profiler

        time_profiler = line_profiler.LineProfiler()
        try:
            return time_profiler(func)(*args, **kwargs)
        finally:
            time_profiler.print_stats()

    return wrapper


def profileit(func):
    def wrapper(*args, **kwargs):
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        ps = pstats.Stats(prof).sort_stats("cumtime")
        ps.print_stats()
        return retval

    return wrapper


def cProfile_decorator(sort_by: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            try:
                return func(*args, **kwargs)
            finally:
                pr.disable()
                pr.print_stats(sort=sort_by)

        return wrapper

    return decorator


# This function reload the module
def reload_module(module):
    import importlib
    importlib.reload(module)
