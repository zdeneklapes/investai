###############################################################################
# TODO: Start here
###############################################################################
import sys

from ai.main import main as ml_main
from ai.shared.exitcode import ExitCode

if __name__ == '__main__':
    ml_main(args={'stock': False})
    sys.exit(ExitCode.OK)
