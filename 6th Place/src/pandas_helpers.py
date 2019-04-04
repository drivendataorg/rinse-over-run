import re
import numpy as np
import pandas as pd

def reduce_memory_usage(df):
    mem0 = df.memory_usage().sum() / 1024**2 
    print(f"memory usage (before): {mem0:.1f} MB")

    for c in df.columns:
        t = str(df[c].dtype)
        new_t = None
        if re.match('float64', t):
            new_t = 'float32'
        elif re.match('^int', t):
            min_v, max_v = df[c].min(), df[c].max()
            for cand in [np.int8, np.int16, np.int32]:
                if min_v >= np.iinfo(cand).min and max_v <= np.iinfo(cand).max:
                    new_t = cand
                    break

        if new_t is not None:
            print(f" {c} {t} -> {new_t}")
            df[c] = df[c].astype(new_t)

    mem1 = df.memory_usage().sum() / 1024**2 
    print(f"memory usage (after): {mem1:.1f} MB")
    return df
