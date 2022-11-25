# -*- coding: utf-8 -*-
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
from pathlib import Path

from dotenv import load_dotenv

# Append root path, because importing from root directory
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()


def parse_cli_argument() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-hp", "--hyperparams", nargs="?", dest="hyperparams", type=str, required=False)
    parser.add_argument("-s", "--stock", dest="stock", action="store_true", required=False)
    return vars(parser.parse_args())


# main
if __name__ == "__main__":
    args = parse_cli_argument()

    # if args["stock"]:
    #     print(f"{args['stock']=}")
    sys.exit()
