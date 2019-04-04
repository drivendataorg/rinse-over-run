import os
import re
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
import hashlib


def ensure_dir(f):
    # from http://stackoverflow.com/questions/273192/python-best-way-to-create-directory-if-it-doesnt-exist-for-file-write
    d = os.path.dirname(f)
    if d != '' and not os.path.exists(d):
        os.makedirs(d)


def log_to_file(log_file):
    ensure_dir(log_file)
    h = logging.FileHandler(log_file)
    h.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
    logging.getLogger().addHandler(h)


def tolist(x):
    return x if isinstance(x, list) else [x]


def hash_of_numpy_array(x):
    """Works only if there are no objects"""
    h = hashlib.sha224(x.tobytes()).hexdigest()
    return h


def hash_of_pandas_df(x):
    s = pd.util.hash_pandas_object(x)
    assert len(s) == len(x)
    return hash_of_numpy_array(s.values)


# from: https://code.i-harness.com/en/q/104706
def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                  # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])


def compress_bool_seq(x, normalize=True, normalize_length=50):
    if normalize:
        x = x.map(float).rolling(normalize_length).mean().dropna().round().map(int)
    lengths, _, values = rle(x.values)
    if lengths is None:
        return ""
    res = []
    for l, v in zip(lengths, values):
        res.append(str(int(v)))
    return "".join(res)


def weighted_median(data, weights, q=0.5):
    """
    Source: https://gist.github.com/tinybike/d9ff1dad515b66cc0d87
    @author Jack Peterson (jack@tinybike.net)
    + slight modifications
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    if len(data)==1:
        return data[0]
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = q * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median