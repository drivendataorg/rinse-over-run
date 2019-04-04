# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def percentile_factory(n):
    """
        Factory which returns an agregation function which compute the n-th percentile
    """

    class Agg:
        def __call__(self, x):
            return np.percentile(x, n)

        def __repr__(self):
            return f"percentile_{n}"

    agg = Agg()
    agg.__name__ = f"percentile_{n}"
    return agg


def last_mean_factory(n):
    """
        Factory which returns an agregation function which compute mean on the n last rows
    """

    class Agg:
        def __call__(self, x):
            return x.tail(n).mean()

        def __repr__(self):
            return f"last_mean_{n}"

    agg = Agg()
    agg.__name__ = f"last_mean_{n}"
    return agg


class ChangesCount:
    def __call__(self, x):
        x = pd.Series(x)
        return (x != x.shift()).sum() - 1

    def __repr__(self):
        return f"changes_count"


class DiffAvg:
    def __call__(self, x):
        x = pd.Series(x)
        return x.diff().mean()

    def __repr__(self):
        return f"diff_avg"


class DiffAbsAvg:
    def __call__(self, x):
        x = pd.Series(x)
        return x.diff().abs().mean()

    def __repr__(self):
        return f"diff_abs_avg"


changes_count = ChangesCount()
changes_count.__name__ = "changes_count"
diff_avg = DiffAvg()
diff_avg.__name__ = "diff_avg"
diff_abs_avg = DiffAbsAvg()
diff_abs_avg.__name__ = "diff_abs_avg"
