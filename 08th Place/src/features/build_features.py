# -*- coding: utf-8 -*-

import logging
import pandas as pd
import time

from src.features.constants import phases


class BaseTransformer:
    def __repr__(self):
        return self.__class__.__name__

    def fit(self, X):
        return self

    def transform(self, X):
        return X.merge(self._transform(X), left_index=True, right_index=True)


class InitTransformer(BaseTransformer):
    """
        Generate a DataFrame with
        - as many rows as process
        - 0 columns
    """
    def transform(self, X):
        return self.values[["process_id"]].drop_duplicates().set_index("process_id")


class InputRecipeTransformer(BaseTransformer):
    """
        Add input_recipe feature
    """
    def __init__(self, data):
        self.data = data

    def _transform(self, X):
        return self.data


class ObservedRecipeTransformer(BaseTransformer):
    """
        Add observed_recipe feature
    """
    def _transform(self, X):
        return (
            self.values.query("phase != 'final_rinse'")[["process_id", "phase"]]
            .drop_duplicates()
            .merge(phases, on="phase")
            .set_index("process_id")
            .drop(["phase"], axis=1)
            .astype(str)
            .groupby(level=0)
            .sum()
            .astype(int)
        )


class ObjectTransformer(BaseTransformer):
    """
        Add object_id feature
    """
    def _transform(self, X):
        return (
            self.values[["process_id", "object_id"]]
            .drop_duplicates()
            .set_index("process_id")
        )


class PipelineTransformer(BaseTransformer):
    """
        Add pipeline feature which is is converted as an int.
    """
    def _transform(self, X):
        df = (
            self.values[["process_id", "pipeline"]]
            .drop_duplicates()
            .set_index("process_id")
        )
        df["pipeline"] = df["pipeline"].apply(lambda p: p[1:]).astype(int)
        return df


class LabelTransformer(BaseTransformer):
    """
        Add label column
    """
    def __init__(self, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = labels

    def transform(self, X):
        return X.merge(self.labels, left_index=True, right_index=True, how="left")


class AggTransformer(BaseTransformer):
    """
        Generate agregations for each subprocess. Takes in input:
        - a list of functions
        - a list of columns on which aggregations should be applied
    """
    def __init__(self, funcnames, cols, verbose=1, by="process_id", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.funcnames = funcnames
        self.cols = [col for col in cols if col != by]
        self.verbose = verbose
        self.by = by

    def transform(self, X):
        dfs = []
        observed_recipes = X["observed_recipe"].unique()
        for it, observed_recipe in enumerate(observed_recipes):
            local_phases = [
                phase["phase"]
                for phase in phases.to_dict("records")
                if str(phase["phase_id"]) in str(observed_recipe)
            ]
            processes = X.query(f"observed_recipe == {observed_recipe}")[
                "process_id"
            ].unique()
            idxv = self.values.process_id.isin(processes) * self.values.phase.isin(
                local_phases
            )

            start = time.time()
            df = (
                self.values.loc[idxv, :]
                .groupby(self.by)[self.cols]
                .agg(self.funcnames)
                .reset_index()
            )
            df["observed_recipe"] = observed_recipe
            dfs.append(df)
            duration = "%.2f" % (time.time() - start)
            if self.verbose >= 1:
                logging.info(
                    f"[Ts Encoder] [{it+1}/{len(observed_recipes)}] duration={duration}s"
                )
        return X.merge(pd.concat(dfs), on=["process_id", "observed_recipe"])


class SimulationTransformer(BaseTransformer):
    """
        Generate subprocess with adjusted weight. For each line in input X
        matrix, it generate k lines with k being the count of steps
        performed by the process of the line.
    """
    def __init__(self, simulate):
        self.simulate = simulate

    def simulators(self):
        return pd.DataFrame(
            data=[
                [1234, 1, .1],
                [1234, 12, .3],
                [1234, 123, .3],
                [1234, 1234, .3],
                [234, 2, .333],
                [234, 23, .333],
                [234, 234, .333],
                [12, 1, .1],
                [12, 12, .9],
            ]
            if self.simulate
            else [],
            columns=["phase_id", "observed_recipe", "weight"],
        )

    def set_simulate(self, simulate):
        self.simulate = simulate

    def transform(self, X):
        ret = X.reset_index().merge(self.simulators(), on="phase_id", how="left")
        ret["weight"] = ret["weight"].fillna(1)
        idx = pd.isnull(ret["observed_recipe"])
        ret.loc[idx, "observed_recipe"] = ret.loc[idx, "phase_id"]
        ret["observed_recipe"] = ret["observed_recipe"].astype(int)
        return ret


class ColumnNameTransformer(BaseTransformer):
    """
        Rename columns to string types as pandas generates columns which name
        is a tuple when aggregating
    """
    def transform(self, X):
        def rename(col):
            return col if isinstance(col, str) else "|".join(col)

        return X.rename(columns={col: rename(col) for col in X.columns})


class Pipeline:
    """
        This class apply in series a list of transformers.
    """
    def __init__(self, encoders):
        self.encoders = encoders

    def attach(self, values):
        for encoder in self.encoders:
            encoder.values = values

    def fit(self, X, y):
        return self

    def transform(self, X):
        durations = []
        for encoder in self.encoders:
            start = time.time()
            X = encoder.transform(X)
            # print(X)
            duration = time.time() - start
            durations.append(duration)
            rows = len(X)
            cols = len(X.columns)
            logging.info(
                "Transform using %(encoder)s in %(duration).2f seconds, final size=%(rows)sx%(cols)s"
                % locals()
            )
        duration = sum(durations)
        logging.info("Total transformation in %(duration).2f seconds" % locals())
        return X
