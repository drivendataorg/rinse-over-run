import re
import logging
from functools import partial
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from src.contest_common import contest_loss
from src.models.common import CommonModel, LgbModel, KerasModel, SegmentationLayer, \
    KFoldLayer, \
    MeanModel, MedianModel, ZeroModel, LinearRegressionModel, BayesianRidgeModel

from src.common import hash_of_pandas_df, weighted_median
from src.keras_utils import generate_simple_model, keras_slice_layer, keras_set_random_state
from src.features.build_features import calc_target_mean_features

class ContestAbstractModel(object):

    def __init__(self, *args, **kwargs):
        pass

    def loss(self, y_pred, y_true, w):
        return np.mean(np.abs(y_pred - y_true) * w)

    def list_features(self, train):
        sel_features = CommonModel.list_features(self, train)
        cnt = train[sel_features].fillna(-9999).nunique()
        sel_features = cnt[cnt>1].index.tolist()
        logging.info(f"list_features={sel_features}")
        return sel_features

    def plot_validation_results(self, df, filename=None):
        filename = filename or "/tmp/validation-results.png"
        assert all(df['target'].notnull())
        y_pred = self.predict(df)
        idx = df.sort_values(by='target').index

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.set_yscale("log", nonposy='clip')
        for s in [y_pred, df['target']]:
            x = range(len(s))
            y = s.reindex(idx).values
            ax.plot(x, y, '-', alpha=0.9)
        ax.legend(['prediction', 'target'])
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

class ContestMeanModel(MeanModel, ContestAbstractModel):
    pass

class ContestMedianModel(MedianModel, ContestAbstractModel):

    def calc_value(self, train):
        median = train['target'].median()
        y_pred = pd.Series(median, index=train.index)
        y_true = train['target']
        w = train['weight']
        f = lambda alpha: ContestAbstractModel.loss(self, y_pred * alpha, y_true, w)

        best_alpha = minimize(f, 1, bounds=[(0, 2)]).x[0]
        logging.info(f"ContestMedianModel median={median:.2f} best_alpha={best_alpha:.4f}")
        return median * best_alpha

class ContestSegmentedMedianModel(ContestAbstractModel):

    def __init__(self, segment_by='object_id', *args, **kwargs):
        self.segment_by = segment_by
        self.models = {}

    def train(self, train, validate=None, test=None):
        for key, g in train.groupby(self.segment_by):
            m = ContestMedianModel()
            m.train(g)
            self.models[key] = m
        m = ContestMedianModel()
        m.train(train)
        self.models['default'] = m

        if validate is not None:
            return self.loss(self.predict(validate), validate['target'], validate['weight'])
        
    def predict(self, df):
        res = []
        for key, g in df.groupby(self.segment_by):
            if key not in self.models:
                key = 'default'
            res.append(self.models[key].predict(g))
        return pd.concat(res, axis=0).reindex(df.index)

class ContestZeroModel(ZeroModel, ContestAbstractModel):
    pass

class ContestLgbModel(LgbModel, ContestAbstractModel):

    def __init__(self, learning_rate=0.05, *args, **kwargs):
        LgbModel.__init__(self, learning_rate=learning_rate, *args, **kwargs)

    def list_features(self, train):
        return ContestAbstractModel.list_features(self, train)


class ContestKerasModel(KerasModel, ContestAbstractModel):

    def __init__(self, layers_num=2, network_size=32, *args, **kwargs):
        self.layers_num = layers_num
        self.network_size = network_size
        KerasModel.__init__(self, *args, **kwargs)

    def list_features(self, train):
        return ContestAbstractModel.list_features(self, train)

    def gen_model(self, feat, output_size):
        assert output_size == 1
        return generate_simple_model(len(feat), 1, layers_num=self.layers_num, network_size=self.network_size)

class ContestLinearRegressionModel(LinearRegressionModel, ContestAbstractModel):
    pass

class ContestBayesianRidgeModel(BayesianRidgeModel, ContestAbstractModel):
    pass

class ContestKerasNaiveModel(ContestKerasModel):

    def gen_model(self, feat, output_size):
        assert output_size == 1
        return generate_simple_model(len(feat), 1, layers_num=0)

class ContestKerasModelV1(ContestKerasModel):

    def __init__(self, layers_num=2, network_size=64, size_per_block=16, dropout=0.2, *args, **kwargs):
        self.size_per_block = size_per_block
        self.dropout = dropout
        super().__init__(layers_num=layers_num, network_size=network_size, *args, **kwargs)

    def gen_model(self, feat, output_size):
        keras_set_random_state(0)
        inputs = keras.layers.Input(shape=(len(feat), ), name='input')
        x = []
        for s in sorted(set(map(lambda x: re.sub(r'^(f_phase_\d)_(.*)$', r'\1', x), feat))):
            if s in feat:
                x.append(keras_slice_layer(inputs, feat, sel_f=[s]))
            else:
                y = keras_slice_layer(inputs, feat, pattern=f'^{s}_.*$')
                y = keras.layers.Dense(self.size_per_block, activation='relu')(y)
                x.append(y)

        if len(x) > 1:
            x = keras.layers.concatenate(x)
        else:
            x = x[0]

        for layer_num in range(self.layers_num):
            x = keras.layers.Dense(self.network_size, activation='relu')(x)
            x = keras.layers.Dropout(self.dropout, seed=layer_num)(x)

        outputs = keras.layers.Dense(output_size, activation='linear')(x)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_absolute_error')
        model.summary()
        return model

class ContestKerasModelV2(ContestKerasModelV1):

    def __init__(self, layers_num=2, network_size=128, size_per_block=24, *args, **kwargs):
        self.size_per_block = size_per_block
        super().__init__(layers_num=layers_num, network_size=network_size, 
            size_per_block=size_per_block, *args, **kwargs)

class ContestKerasModelV3(ContestKerasModelV1):

    def __init__(self, layers_num=2, network_size=128, size_per_block=24, *args, **kwargs):
        self.size_per_block = size_per_block
        super().__init__(layers_num=layers_num, network_size=network_size, 
            size_per_block=size_per_block, *args, **kwargs)

    def gen_model(self, feat, output_size):
        keras_set_random_state(0)
        inputs = keras.layers.Input(shape=(len(feat), ), name='input')

        m1 = super().gen_model(feat, output_size)
        m2 = generate_simple_model(len(feat), 1, layers_num=1, network_size=16, dropout=0.1)

        x = keras.layers.concatenate([
            m1(inputs),
            m2(inputs)
        ])

        outputs = keras.layers.Dense(output_size, activation='linear')(x)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_absolute_error')
        model.summary()
        return model


class ContestBlendedModel(ContestAbstractModel):

    def __init__(self, models_factory=None, strategy='mean', *args, **kwargs):
        logging.info("ContestBlendedModel::__init__")
        assert strategy in ['mean', 'LinearRegression', 'NN', 'OptMix']
        if models_factory is None:
            models_factory = [ContestLgbModel, ContestKerasModelV1]
        self.models_factory = models_factory
        self.models = []
        self.strategy = strategy
        self.args = args
        self.kwargs = kwargs
        ContestAbstractModel.__init__(self, *args, **kwargs)

    def strategy_train_lr(self, train, valid):
        m = LinearRegression(fit_intercept=False)
        y_preds = self._predict(train)
        m.fit(y_preds.values, train['target'], train['weight'])
        logging.info(f"LinearRegression.coef_={m.coef_}")
        self.strategy_model = m

    def strategy_train_opt_mix(self, train, valid):
        df = pd.concat([train.iloc[:50], valid], axis=0)
        y_preds = self._predict(df)
        num = y_preds.shape[1]

        m = LinearRegression(fit_intercept=False)
        m.intercept_ = 0.0

        best_loss = None
        best_coef = None
        for alpha in np.linspace(0.0, 1.0, 101):
            beta = (1 - alpha) / (num - 1)
            coef_ = np.array([alpha] + [beta] * (num - 1))
            assert abs(sum(coef_) - 1.0) < 1e-6
            y_pred = np.sum(np.multiply(y_preds, coef_), axis=1)
            y_pred = pd.Series(y_pred, index=df.index)
            loss = self.loss(y_pred, df['target'], df['weight'])
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_coef = coef_

        m.coef_ = best_coef
        logging.info(f"OptMix.coef_={m.coef_} best_loss={best_loss:.4f}")
        self.strategy_model = m

    def strategy_train_nn(self, train, valid):
        m = LinearRegression(fit_intercept=False)
        y_preds = self._predict(train)
        keras_set_random_state(0)
        m = generate_simple_model(y_preds.shape[1], 1, layers_num=1, network_size=1, dropout=0)
        m.fit(
            y_preds.values,
            train['target'].values,
            sample_weight=train['weight'].values,
            epochs=100,
            batch_size=1000
        )
        self.strategy_model = m


    def train(self, train, valid=None, test=None):
        logging.info("ContestBlendedModel::train")
        self.models = [m(*self.args, **self.kwargs) for m in self.models_factory]
        cv_from_models = [m.train(train, valid, test) for m in self.models]

        if self.strategy == 'LinearRegression':
            self.strategy_train_lr(train, valid)
        elif self.strategy == 'OptMix':
            self.strategy_train_opt_mix(train, valid)
        elif self.strategy == 'NN':
            self.strategy_train_nn(train, valid)
        if valid is None:
            return None
        else:
            y_pred = self.predict(valid)
            cv = self.loss(y_pred, valid['target'], valid['weight'])
            logging.info(f"cv from models: {cv_from_models}, blended cv: {cv:.4f}")
            return cv

    def strategy_predict(self, y_preds):
        if self.strategy == 'mean':
            return y_preds.mean(axis=1).values
        elif self.strategy in ['LinearRegression', 'OptMix']:
            return self.strategy_model.predict(y_preds.values)
        elif self.strategy in ['NN']:
            return self.strategy_model.predict(y_preds.values)[:, 0]
        else:
            raise Exception("not supported")

    def _predict(self, df):
        y_preds = [m.predict(df).rename(i) for i, m in enumerate(self.models)]
        y_preds = pd.concat(y_preds, axis=1)
        return y_preds

    def predict(self, df):
        y_preds = self._predict(df)
        return pd.Series(self.strategy_predict(y_preds), index=df.index, name="pred")


class ContestBlendedModelV1(ContestBlendedModel):

    def __init__(self, epochs=100, random_state=0, patience=50, *args, **kwargs):
        models_factory = []
        models_factory.append(partial(ContestLgbModel, learning_rate=0.08, random_state=random_state, early_stopping=20, num_leaves=32))
        models_factory.append(partial(ContestKerasModelV1, network_size=128, size_per_block=16,
            epochs=epochs, batch_size=100, patience=patience, random_state=random_state))

        super().__init__(strategy='OptMix', models_factory=models_factory)

class ContestSegmentationLayer(SegmentationLayer, ContestAbstractModel):

    limit_predictions = False

    def filter_features(self, df, max_phase_num):
        sel_columns = []
        for c in sorted(df.columns):
            m = re.match(r'f_phase_(\d)_', c)
            if m is None or int(m.group(1)) <= max_phase_num:
                sel_columns.append(c)
        return df[sel_columns]

    def segment(self, train, validate, test):
        assert test is not None
        for (e, mx), seg_test in test.groupby([
                test['expected_phase_summary'], 
                test['phase_summary'].str.rstrip("?").str.rstrip("0").map(len).clip(1, 4)
            ]):
            f = lambda x: x[(x.expected_phase_summary==e) & (x.max_phase_num>=mx)]
            g = lambda x: self.filter_features(x, mx)
            seg_train = g(f(train)) if train is not None else None
            seg_validate = g(f(validate)) if validate is not None else None
            seg_test = g(seg_test)
            yield ((e, mx), len(seg_test) / len(test)), (seg_train, seg_validate, seg_test)

    def train(self, train, validate=None, test=None):
        assert all(train['target'].notnull())
        f = lambda x: train.groupby(x)['target']
        self.targets_by_object = {
            'min': f('object_id').min(),
            'max': f('object_id').max()
        }
        self.targets_by_pipeline = {
            'min': f('pipeline').min(),
            'max': f('pipeline').max()
        }
        self.targets = {
            'min': train['target'].min(),
            'max': train['target'].max()
        }
        return SegmentationLayer.train(self, train, validate, test)

    def predict(self, test):
        res = SegmentationLayer.predict(self, test)
        f = lambda x: test['object_id'].map(self.targets_by_object[x])\
                .fillna(test['pipeline'].map(self.targets_by_pipeline[x]))\
                .fillna(self.targets[x])\
                .rename(x)
        tmp = pd.concat([
            res.rename('pred'),
            f('min'), f('max')
        ], axis=1)
        if self.limit_predictions:
            tmp['pred'] = np.clip(tmp['pred'], tmp['min'], tmp['max'])
        assert all(tmp['pred'].notnull())
        return tmp['pred']

class ContestKFoldLayer(KFoldLayer, ContestAbstractModel):

    shuffle = False
    recompute_target_mean = False

    def detect_max_phase(self, df):
        return max(
            int(x.split("_")[2]) for x in df.filter(regex='f_phase_\d').columns.tolist()
        )

    def split_train(self, train, trn_idx, val_idx):
        if self.recompute_target_mean:
            max_phase_num = self.detect_max_phase(train)
            logging.info(f"ContestKFoldLayer recomputing target_mean_features max_phase_num={max_phase_num}")
            train = calc_target_mean_features(train, trn_idx, max_phase_num=max_phase_num)
            sel = train.loc[val_idx]
            logging.info(f"dates: {sel.timestamp_min.min().strftime('%Y-%m-%d')} - {sel.timestamp_min.max().strftime('%Y-%m-%d')}")
        train = train.copy()
        train['target'] = train['target'].clip(-1, 1)
        return KFoldLayer.split_train(self, train, trn_idx, val_idx)

    def train(self, train, validate=None, test=None):
        train = train.sort_values(by='timestamp_min')

        if self.recompute_target_mean:
            max_phase_num = self.detect_max_phase(train)
            tmp = pd.concat([train, test], axis=0).sort_values(by='timestamp_min')
            tmp = calc_target_mean_features(tmp, None, max_phase_num=max_phase_num)
            self.saved_test = tmp[~tmp.is_train]

        return KFoldLayer.train(self, train, validate, test)

    def predict(self, df):
        if self.recompute_target_mean and any(df.index.isin(self.saved_test.index)):
            assert all(df.index.isin(self.saved_test.index))
            res = KFoldLayer.predict(self, self.saved_test)
            logging.info("overriding test features with saved values")
            return res.reindex(df.index)
        else:
            return KFoldLayer.predict(self, df)



class ContestScalingLayer(object):

    def __init__(self, model_factory=None, model_kwargs=None, limit_targets=True, *args, **kwargs):
        print(f"in ContestScalingLayer init args={args} kwargs={kwargs}")
        super().__init__()
        self.model_factory = model_factory
        self.model_kwargs = model_kwargs or {}
        self.limit_targets = limit_targets

    @staticmethod
    def scale_target(train, res):
        res['target_shift'] = res['f_phase_1_target_median_0.50']
        res['target_scale'] = (
            res['f_phase_1_target_median_0.50'] - res['f_phase_1_target_median_0.01']
        ).clip(10**3, None)
        return res


    @staticmethod
    def scale(df, train, sel_features=None):
        if df is None:
            return None
        if sel_features is None:
            sel_features = df.filter(regex='^f_').columns.tolist()

        res = df.copy()
        
        res = ContestScalingLayer.scale_target(train, res)

        res['target'] = (res['target'].astype('float64') -  res['target_shift']) / res['target_scale']
        res['weight'] = res['weight'].astype('float64') * res['target_scale']

        # return res

        from sklearn.preprocessing import StandardScaler
        for c in sel_features:
            if train[c].abs().max() > 1:
                if True:
                    x = train[c]
                    # x = x.append(df[c])
                    s = StandardScaler().fit(x.astype('float64').values.reshape(-1, 1))
                    res[c] = res[c].astype('float64')
                    res[c] = s.transform(res[c].values.reshape(-1, 1))[:, 0]
                    # res[c] = res[c].clip(-32, 32)
                    continue
                shift = train[c].median()
                scale = shift
                res[c] = ((res[c].astype('float64') - shift) / scale).clip(-10, 10)
        
        return res

    @staticmethod
    def descale_pred(pred, df):
        return ((pred * df['target_scale']) + df['target_shift']).clip(0, None)

    def train(self, train, validate=None, test=None):
        logging.info("inside scaling train")
        logging.info("ContestScalingLayer::train hash(train)={}".format(hash_of_pandas_df(train)))
        self.model = self.model_factory(**self.model_kwargs)

        self.org_train = train.copy()

        train = ContestScalingLayer.scale(train, self.org_train)
        validate = ContestScalingLayer.scale(validate, self.org_train)
        test = ContestScalingLayer.scale(test, self.org_train)
        cv = self.model.train(train, validate, test)
        return cv

    def predict(self, df):
        df_scaled = ContestScalingLayer.scale(df, self.org_train)
        res = self.model.predict(df_scaled)
        if self.limit_targets:
            logging.info("ContestScalingLayer::predict clipping results")
            res = res.clip(-1, 1)

        return ContestScalingLayer.descale_pred(res, df_scaled)


class ContestBlendedModelV2(ContestBlendedModel):

    def __init__(self, *args, **kwargs):
        models_factory = []
        models_factory.append(partial(ContestSegmentedMedianModel, segment_by='object_id'))
        models_factory.append(partial(ContestSegmentedMedianModel, segment_by='pipeline'))

        super().__init__(strategy='OptMix', models_factory=models_factory)


class ContestBlendedModelV3(ContestBlendedModel):

    def __init__(self, epochs=100, random_state=0, *args, **kwargs):
        models_factory = []
        models_factory.append(partial(ContestLgbModel, learning_rate=0.08, random_state=random_state, early_stopping=20, num_leaves=32))
        models_factory.append(partial(ContestKerasModelV2, \
            layers_num=2, network_size=256, size_per_block=16,
            epochs=epochs, batch_size=100, patience=50, random_state=random_state))

        super().__init__(strategy='OptMix', models_factory=models_factory)


class ContestBlendedModelV3(ContestBlendedModel):

    def __init__(self, epochs=100, random_state=0, patience=50, *args, **kwargs):
        models_factory = []
        models_factory.append(partial(ContestLgbModel, learning_rate=0.08, random_state=random_state, early_stopping=20, num_leaves=32))
        models_factory.append(partial(ContestKerasModelV3, \
            layers_num=2, network_size=128, size_per_block=16,
            epochs=epochs, batch_size=100, patience=patience, random_state=random_state))

        super().__init__(strategy='OptMix', models_factory=models_factory)
