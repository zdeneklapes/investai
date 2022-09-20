import distutils.version
import os
import sys
import warnings

from gym import error, logger, spaces, wrappers
from gym.core import (
    ActionWrapper,
    Env,
    GoalEnv,
    ObservationWrapper,
    RewardWrapper,
    Space,
    Wrapper,
)
from gym.envs import make, spec
from gym.utils import reraise
from gym.version import VERSION as __version__


def undo_logger_setup():
    warnings.warn(
        "gym.undo_logger_setup is deprecated. gym no longer modifies the global logging configuration"
    )


__all__ = ["Env", "Space", "Wrapper", "make", "spec", "wrappers"]
