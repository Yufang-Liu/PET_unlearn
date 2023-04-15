import os
from os import path
import random, torch
import numpy as np
import collections, toml

# python 3.8+ compatibility
try:
    collectionsAbc = collections.abc
except:
    collectionsAbc = collections


def set_seed(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update(d, u):
    for k, v in u.items():
        dv = d.get(k, {})
        if not isinstance(dv, collectionsAbc.Mapping):
            d[k] = v
        elif isinstance(v, collectionsAbc.Mapping):
            d[k] = update(dv, v)
        else:
            d[k] = v
    return d


def load_config(config_path):
    print("reading config from <{}>\n".format(path.abspath(config_path)))
    try:
        with open(config_path, "r") as f:
            config = toml.load(f)
            return config
    except FileNotFoundError as e:
        print("can not find config file")
        raise e

