import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit


class ProcessSplit(PredefinedSplit):
    """
        Custom sklearn.PredefinedSplit which make sure that subprocess which
        belong to the same split (train or test)
    """

    def __init__(self, seed, k, tf):
        x = tf.process_id.unique()
        np.random.RandomState(seed=seed).shuffle(x)
        df = pd.DataFrame({"process_id": x}).reset_index()
        df["index"] = df["index"].mod(k)
        super().__init__(test_fold=tf.merge(df, on="process_id")["index"])
