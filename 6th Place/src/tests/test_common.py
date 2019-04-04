import numpy as np
import pandas as pd
from src.common import hash_of_numpy_array


def test_hash_of_numpy_array():
    h1 = hash_of_numpy_array(np.array([1, 2, 3]))
    h2 = hash_of_numpy_array(np.array([1, 2, 3]))
    h3 = hash_of_numpy_array(np.array([1, 2, 1]))
    assert h1 == h2
    assert h1 != h3

def test_hash_of_numpy_array_ver2():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    df2 = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    h1 = hash_of_numpy_array(df.values)
    h2 = hash_of_numpy_array(df.values)
    h3 = hash_of_numpy_array(df2.values)
    assert h1 == h2
    assert h1 == h3