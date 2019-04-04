import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, BayesianRidge
import lightgbm as lgb
from src.common import hash_of_numpy_array, hash_of_pandas_df
from src.keras_utils import generate_simple_model, keras_set_random_state, keras_hash_of_model
import keras
from sklearn.model_selection import KFold, LeaveOneGroupOut


class CommonModel(object):
    target_column = "target"
    weight_column = "weight"

    def __init__(self, random_state=0, label=None, *args, **kwargs):
        self.random_state = random_state
        self.label = label

    def list_features(self, df):
        return df.filter(regex='^f_').columns.tolist()

    def prepare_model_data(self, df):
        return df[self.features]


class ConstantPredictionModel(CommonModel):

    def train(self, train, valid=None, test=None):
        self.value = self.calc_value(train)
        logging.info(f"constant prediction model, calculated value={self.value:.2f}")

        if valid is not None:
            valid_loss = self.loss(
                y_pred=self.predict(valid), y_true=valid[self.target_column].values,
                w=valid[self.weight_column].values
            )
        else:
            valid_loss = None
        return valid_loss

    def predict(self, df):
        return pd.Series(self.value, index=df.index)

class SciKitModel(CommonModel):

    def prepare_model_data(self, df):
        return super().prepare_model_data(df).fillna(0)

    def gen_model(self):
        raise Exception("not implemeted")        
        self.model = LinearRegression()

    def train(self, train, valid=None):
        self.features = self.list_features(train)
        self.model = self.gen_model()
        self.model.fit(
            self.prepare_model_data(train).values,
            train[self.target_column].values,
            train[self.weight_column].values
        )

        if valid is not None:
            valid_loss = self.loss(
                y_pred=self.predict(valid), y_true=valid[self.target_column].values,
                w=valid[self.weight_column].values
            )
        else:
            valid_loss = None
        return valid_loss

    def predict(self, df):
        res = self.model.predict(self.prepare_model_data(df))
        return pd.Series(res, index=df.index)

class LinearRegressionModel(SciKitModel):

    def gen_model(self):
        return LinearRegression()

class BayesianRidgeModel(SciKitModel):

    def gen_model(self):
        return BayesianRidge()

class MeanModel(ConstantPredictionModel):

    def calc_value(self, df):
        assert all(df[self.target_column].notnull())
        return np.mean(df[self.target_column])

class MedianModel(ConstantPredictionModel):

    def calc_value(self, df):
        assert all(df[self.target_column].notnull())
        return np.median(df[self.target_column])

class ZeroModel(ConstantPredictionModel):

    def calc_value(self, df):
        assert all(df[self.target_column].notnull())
        return 0


class LgbModel(CommonModel):
    
    def __init__(self, params=None, random_state=0, learning_rate=0.1, 
            early_stopping=20,
            min_data_in_leaf=30, num_boost_round=500, label=None, *args, **kwargs):
        super().__init__(label=label, random_state=random_state, *args, **kwargs)
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'min_data_in_leaf': min_data_in_leaf,
            'metric': 'mae',
            'learning_rate': learning_rate,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'random_state': random_state,
            'num_boost_round': num_boost_round,
            'early_stopping': early_stopping,
            'verbose': -1
        }
        if params is not None:
            self.params.update(params)
        return super().__init__(*args, **kwargs)

    def loss(self, y_pred, y_true, w):
        assert len(y_pred) == len(y_true)
        return np.mean(np.abs(y_pred - y_true) * w)

    def train(self, train, valid=None, test=None):
        np.random.seed(self.params.get('random_state', 0))

        self.features = self.list_features(train)
        logging.info("training using %d features, train_size=%d valid_size=%d",
            len(self.features), len(train), len(valid)
        )
        dtrain = lgb.Dataset(
            self.prepare_model_data(train).values, 
            train[self.target_column].values, 
            weight=train[self.weight_column], feature_name=self.features,
            free_raw_data=False)
        dvalid = lgb.Dataset(self.prepare_model_data(valid).values,
            valid[self.target_column].values, 
            weight=valid[self.weight_column], feature_name=self.features, 
            reference=dtrain, free_raw_data=False)

        def custom_error(preds, train_data):
            labels = train_data.get_label()
            weights = train_data.get_weight()
            return 'error', np.mean(np.abs(preds - labels) * weights), False

        logging.info("lgb.train params=%s", self.params)
        np.random.seed(self.params.get('random_state', 0))
        self.model = lgb.train(self.params, dtrain, 
                  num_boost_round=self.params.get('num_boost_round'), 
                  valid_sets=(dtrain, dvalid), 
                  valid_names=('train', 'valid',), 
                  verbose_eval=20, 
                  feval=custom_error,
                  early_stopping_rounds=self.params.get('early_stopping'))

        self.feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importance()
        })

        valid_loss = self.loss(
            y_pred=self.predict(valid), y_true=valid[self.target_column].values,
            w=valid[self.weight_column].values
        )

        return valid_loss

    def predict(self, df):
        return pd.Series(self.model.predict(
            self.prepare_model_data(df).values,
            num_iteration=self.model.best_iteration
        ), index=df.index)


class KerasModel(CommonModel):
    target_column = "target"
    weight_column = "weight"
    
    def __init__(self, epochs=200, batch_size=32, output_size=1, patience=50, params=None, random_state=0, label=None, *args, **kwargs):
        super().__init__(label=label, *args, **kwargs)
        self.output_size = output_size
        self.params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'random_state': random_state,
            'patience': patience
        }
        if params is not None:
            self.params.update(params)
        logging.info(self.params)
        return super().__init__(*args, **kwargs)

    def loss(self, y_pred, y_true, w):
        assert len(y_pred) == len(y_true)
        return np.mean(np.abs(y_pred - y_true) * w)

    def gen_model(self, feat, output_size=1):
        return generate_simple_model(len(feat), output_size)

    def prepare_model_data(self, df):
        res = super().prepare_model_data(df).astype('float64').fillna(0.0)
        return res

    def train(self, train, valid=None, test=None):
        self.features = self.list_features(train)
        logging.info("KerasModel::train hash(train)={}".format(hash_of_pandas_df(train)))
        logging.info("KerasModel::train hash(train_data)={}".format(hash_of_pandas_df(self.prepare_model_data(train))))

        logging.info("training using %d features, train_size=%d valid_size=%d",
            len(self.features), len(train), len(valid)
        )
        validation_data = None
        if valid is not None:
            validation_data = (
                self.prepare_model_data(valid).values,
                valid[self.target_column].values,
                valid[self.weight_column].values
            )
        np.random.seed(0)
        keras_set_random_state(self.params.get('random_state', 0))
        self.model = self.gen_model(self.features, self.output_size)
        logging.info("KerasModel::train hash(model before training)={}".format(keras_hash_of_model(self.model)))
        callbacks = []
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.params.get('patience')))
        callbacks.append(keras.callbacks.ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=0, save_best_only=True))
        keras_set_random_state(self.params.get('random_state', 0))
        logging.info("params: {}".format(self.params))
        self.model.fit(
            self.prepare_model_data(train).values,
            train[self.target_column].values,
            sample_weight=train[self.weight_column].values,
            validation_data=validation_data,
            epochs=self.params.get('epochs'),
            batch_size=self.params.get('batch_size'),
            shuffle=True,
            verbose=2,
            callbacks=callbacks
        )
        self.model.load_weights('/tmp/weights.hdf5')
        valid_loss = self.loss(
            y_pred=self.predict(valid).values, y_true=valid[self.target_column].values,
            w=valid[self.weight_column].values
        )
        logging.info("KerasModel::train hash(model after training)={}".format(keras_hash_of_model(self.model)))

        return valid_loss

    def predict(self, df):
        res = self.model.predict(self.prepare_model_data(df).values)
        if self.output_size == 1:
            return pd.Series(res[:, 0], index=df.index)
        else:
            return pd.DataFrame(res, index=df.index)



class SegmentationLayer(object):

    def __init__(self, model_factory, model_kwargs=None, *args, **kwargs):
        super().__init__()
        self.model_factory = model_factory
        self.model_kwargs = model_kwargs or {}
        self.models = {}

    def segment(self, train, validate, test):
        return {"default": (train, valid, test)}

    def train(self, train, validate=None, test=None):
        logging.info("SegmentationLayer::train hash={}".format(hash_of_pandas_df(train)))
        cv = []
        total_cv = 0.0
        for (seg_label, seg_weight), (seg_train, seg_validate, seg_test) in self.segment(train, validate, test):
            print(f"calling model_factory={self.model_factory} with kwargs={self.model_kwargs}")
            model = self.model_factory(label=seg_label, **self.model_kwargs)
            logging.info(f"training segment {seg_label}:"
                f" train={seg_train.shape if seg_train is not None else None}"
                f" validate={seg_validate.shape if seg_validate is not None else None}"
                f" test={seg_test.shape if seg_test is not None else None}"
            )
            seg_cv = model.train(seg_train, seg_validate)
            cv.append({
                'label': seg_label,
                'weight': seg_weight,
                'loss': seg_cv,
                'weighted_loss': seg_cv * seg_weight
            })
            total_cv += seg_cv * seg_weight
            self.models[seg_label] = model
        self.cv = pd.DataFrame(cv, columns=['label', 'loss', 'weight', 'weighted_loss'])\
            .sort_values(by='label')
        return total_cv

    
    def predict(self, test):
        res = pd.Series(index=test.index)
        for (seg_label, seg_weight), (_, _, seg_test) in self.segment(None, None, test):
            seg_pred = self.models[seg_label].predict(seg_test)
            res.loc[seg_pred.index] = seg_pred
        return res


class KFoldLayer(object):
    """actually it is LeaveOneGroupOut"""

    shuffle = True

    def __init__(self, model_factory, model_kwargs=None, n_splits=8, *args, **kwargs):
        super().__init__()
        self.model_factory = model_factory
        self.model_kwargs = model_kwargs or {}
        self.models = []
        self.n_splits = n_splits

    def split_train(self, train, trn_idx, val_idx):
        train_part = train.loc[trn_idx]
        validate_part = train.loc[val_idx]
        return train_part, validate_part

    # def loss(y_pred, y_true, w):
    #    raise Exception("Not implemented")

    def train(self, train, validate=None, test=None):
        assert validate is None
        logging.info("KFoldLayer::train hash={}".format(hash_of_pandas_df(train)))
        folds = LeaveOneGroupOut()
        groups = np.floor(np.linspace(0, 1, len(train), False) * self.n_splits).astype(int)
        y_pred = pd.Series(index=train.index)
        self.models = []
        local_cvs = []
        for fold_, (trn_idx_, val_idx_) in enumerate(folds.split(train.values, groups, groups)):
            trn_idx = train.iloc[trn_idx_].index
            val_idx = train.iloc[val_idx_].index

            logging.info(f"fold {fold_}")
            train_part, validate_part = self.split_train(train, trn_idx, val_idx)
            model = self.model_factory(**self.model_kwargs)
            cv = model.train(train_part, validate_part, test)
            local_cvs.append(cv)
            self.models.append(model)

            logging.info(f"KFoldLayer::train fold {fold_} local cv: {cv:.4f}")
            y_pred.loc[val_idx] = model.predict(validate_part).values
        
        total_cv = self.loss(y_pred, train['target'], train['weight'])
        logging.info(f"KFoldLayer::train total cv: {total_cv:.4f} local_cvs: {local_cvs}")
        return total_cv

    
    def predict(self, test):
        res = pd.Series([0.0] * len(test), index=test.index)
        for model in self.models:
            res += model.predict(test) / len(self.models)
        return res