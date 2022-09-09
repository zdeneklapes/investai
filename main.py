###############################################################################
# TODO: Start here
###############################################################################
import sys

from ml.main import main as ml_main
from ml.shared.exitcode import ExitCode

if __name__ == '__main__':
    ml_main(args={'stock': False})
    sys.exit(ExitCode.OK)
