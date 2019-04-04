import pandas as pd
import numpy as np

T = 290_000

def contest_loss(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true) / np.clip(y_true, T, None))