import os
import json
import random
import bisect
import string
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import perf_counter
from math import isnan
from threading import Lock
from pathlib import Path
from typing import Tuple
from typing import Callable
from typing import Optional
from typing import List
from os.path import basename
from src.utils import append_to_file
from src.utils import guarantee_path_exist


class IdeaSearcher:
    
    # ----------------------------- IdeaSearhcer 初始化 ----------------------------- 
    
    def __init__(
        self
    ) -> None:
        
        pass
    
    # ----------------------------- 外部调用动作 ----------------------------- 
    
    
    # ----------------------------- 内部调用动作 ----------------------------- 