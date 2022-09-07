__author__ = "Zdeněk Lapeš"
__copyright__ = "Copyright 2022, The NameProject"
__credits__ = ["TODO"]
__license__ = "TODO"
__version__ = "TODO"
__maintainer__ = "Zdeněk Lapeš"
__email__ = ["lapes.zdenek@gmail.com"]
__status__ = "Production"

"""
TODO:

IDEAS (MAIN)
    Build                   (Neural network | ...) 
    which will              (train NN | ...)
    that can predict the    (next move | long prediction | ...)
    based on the            (change of any fundamental data | ...).

IDEAS (STOCK'S DIVISION)
    Branch

IDEAS (FEATURES_1: SORT BY)
    Which companies have the most amount of cash available relative to amount of debt?
    Which companies have the most biggest profit relative to amount of debt?
    Which companies have the biggest sales margin?
    Which companies have the biggest sales margin?
    Which companies are increasing profits constantly
    Which companies are decreasing debts constantly?
    Which companies are increasing dividend yield?
    Which companies are the oldest?
    Which companies are the youngest?

    Result: (make the union on the top most companies | train NN | ...)
    
IDEAS (FEATURES_2)
    The company exist a long time so the growth can be slower than in the young company
    
!!!
    Do that as a finite/infinite automatically building state machine consists in probabilities
"""

import argparse
import sys

from dotenv import load_dotenv

from ml.shared.types import param_type
from ml.shared.exitcode import ExitCode

load_dotenv()


def parse_cli_argument() -> dict:
    """
    Parse cli arguments

    Returns:
        dict: cli arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyperparams', nargs='?', dest='hyperparams', type=str, required=True)
    parser.add_argument('-s', '--stock', dest='stock', action='store_true', required=False)
    args = vars(parser.parse_args())
    return args


def main(args: param_type):
    """
    Based on cli arguments run modules

    Args:
        args: all cli arguments
    """

    if args['stock']:
        print(f"{args['stock']=}")


if __name__ == '__main__':
    cli_args = parse_cli_argument()
    main(args=cli_args)
    sys.exit()
