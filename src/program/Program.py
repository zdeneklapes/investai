# -*- coding: utf-8 -*-
import dataclasses

import pandas as pd
from common.Args import Args


@dataclasses.dataclass
class Program:
    args: Args
    dataset: pd.DataFrame
