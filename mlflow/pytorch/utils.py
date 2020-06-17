import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc
import re
tqdm.pandas()
# from fastprogress.fastprogress import master_bar, progress_bar
from pathlib import Path

from contextlib import contextmanager
import time
import string
import warnings
warnings.filterwarnings('ignore')
from collections import Iterable
import random
import os
from sklearn.metrics import roc_auc_score
# from callback_logging import Net
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

@contextmanager
def timer(msg):
    t0 = time.time()
    print(f'[{msg}] start.')
    yield
    elapsed_time = time.time() - t0
    print(f'[{msg}] done in {elapsed_time / 60:.2f} min.')

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def roc(out, y):
    score = roc_auc_score(y.cpu().detach().numpy(), out.cpu().detach().numpy())
    return score
def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb.long()).float().mean()

def sigmoid(x): return 1/(1+np.exp(-x))
