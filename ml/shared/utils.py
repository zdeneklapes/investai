from os import path
from typing import Union, Optional, Dict
import yaml


def parse_yaml(*, filename: str) -> Optional[Union[Dict[str, str], FileNotFoundError]]:
    with open(filename, "r") as stream:
        yaml_parsed = yaml.safe_load(stream)
        yaml_parsed |= yaml_parsed['default']
        del yaml_parsed['default']
        return yaml_parsed


def get_module_root_dirs(*, file_path: str, deep=0):
    module_dir = path.realpath(file_path)
    root_dir = module_dir
    for i in range(deep + 1):
        root_dir = path.dirname(root_dir)
    return root_dir, module_dir
