__author__ = "Zdeněk Lapeš"
__copyright__ = "Copyright 2022, The NameProject"
__credits__ = ["TODO"]
__license__ = "TODO"
__version__ = "TODO"
__maintainer__ = "Zdeněk Lapeš"
__email__ = ["lapes.zdenek@gmail.com"]
__status__ = "Development"

import argparse
import sys

from dotenv import load_dotenv

from ai.shared.types import param_type

load_dotenv()


def parse_cli_argument() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyperparams', nargs='?', dest='hyperparams', type=str, required=True)
    parser.add_argument('-s', '--stock', dest='stock', action='store_true', required=False)
    args = vars(parser.parse_args())
    return args


def main(args: param_type):
    if args['stock']:
        print(f"{args['stock']=}")


if __name__ == '__main__':
    main(args=parse_cli_argument())
    sys.exit()
