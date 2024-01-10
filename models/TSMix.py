import numpy as np
import pandas as pd
import tensorflow as tf
from models.base import Model
from preprocessing.simple_transforms import Logarithm
from preprocessing.advanced_transforms import  LGBM_impute_nan_demand
from sklearn.preprocessing import StandardScaler
import os

RANDOM_SEED = 46

class TSDataLoader:
    
    def __init__(
          self, data, batch_size, seq_len, pred_len
        ):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_slice = slice(0, None)

        self._read_data()

    def _read_data(self):
        df = self.data

        train_df = df
        test_df = df[- 2*self.seq_len : ]

        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.train_df = scale_df(train_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)
        self.n_feature = self.train_df.shape[-1]

    def _split_window(self, data):
        inputs = data[:, : self.seq_len, :]
        labels = data[:, self.seq_len :, self.target_slice]
        inputs.set_shape([None, self.seq_len, None])
        labels.set_shape([None, self.pred_len, None])
        return inputs, labels

    def _make_dataset(self, data, shuffle=True):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=(self.seq_len + self.pred_len),
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=self.batch_size,
        )
        ds = ds.map(self._split_window)
        return ds

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_train(self, shuffle=True):
        return self._make_dataset(self.train_df, shuffle=shuffle)

    def get_test(self):
        return self._make_dataset(self.test_df, shuffle=False)



def res_block(inputs, norm_type, activation, dropout, ff_dim):
    
    norm = (
      tf.keras.layers.LayerNormalization
      if norm_type == 'L'
      else tf.keras.layers.BatchNormalization
    )

    # Temporal Linear
    x = norm(axis=[-2, -1])(inputs)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = tf.keras.layers.Dense(x.shape[-1], activation=activation)(x)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
    x = tf.keras.layers.Dropout(dropout)(x)
    res = x + inputs

    # Feature Linear
    x = norm(axis=[-2, -1])(res)
    x = tf.keras.layers.Dense(ff_dim, activation=activation)(x)  # [Batch, Input Length, FF_Dim]
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
    x = tf.keras.layers.Dropout(dropout)(x)
    return x + res


def build_model(
        input_shape,
        pred_len,
        norm_type,
        activation,
        n_block,
        dropout,
        ff_dim,
        target_slice,
    ):
    
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs  # [Batch, Input Length, Channel]
    rev_norm = RevNorm(axis=-2)
    x = rev_norm(x, 'norm')
    for _ in range(n_block):
        x = res_block(x, norm_type, activation, dropout, ff_dim)

    if target_slice:
        x = x[:, :, target_slice]

    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = tf.keras.layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
    outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])
    outputs = rev_norm(outputs, 'denorm', target_slice)
    return tf.keras.Model(inputs, outputs)


class RevNorm(tf.keras.layers.Layer):
    def __init__(self, axis, eps=1e-5, affine=True):
        super().__init__()
        self.axis = axis
        self.eps = eps
        self.affine = affine

    def build(self, input_shape):
        if self.affine:
            self.affine_weight = self.add_weight(
              'affine_weight', shape=input_shape[-1], initializer='ones'
            )
            self.affine_bias = self.add_weight(
              'affine_bias', shape=input_shape[-1], initializer='zeros'
            )

    def call(self, x, mode, target_slice=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = tf.stop_gradient(
            tf.reduce_mean(x, axis=self.axis, keepdims=True)
        )
        self.stdev = tf.stop_gradient(
            tf.sqrt(
                tf.math.reduce_variance(x, axis=self.axis, keepdims=True) + self.eps
            )
        )

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = x - self.affine_bias[target_slice]
            x = x / self.affine_weight[target_slice]
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x


class TSMix(Model):

    def __init__(self, train_epochs, dropout) -> None:
        super().__init__()
        self.model = None

        self.seq_len = 24 * 7 * 4 # 'input sequence length'
        self.pred_len = 24 * 7 # prediction sequence length'
        self.n_block = 2 # number of block for deep architecture',
        self.ff_dim = 32 # fully-connected feature dimension',
        self.dropout = dropout
        self.norm_type = 'B' # choices=['L', 'B'], 'LayerNorm or BatchNorm',
        self.activation = 'relu' # choices=['relu', 'gelu'],
        self.hidden_dim = 64 # hidden feature dimension'

        self.train_epochs = train_epochs
        self.batch_size = 24
        self.learning_rate = 0.0001
        self.patience = 5 # number of epochs to early stop'


    def fit(self, demands, weather):

        tf.keras.utils.set_random_seed(RANDOM_SEED)

        self.data_loader = TSDataLoader(
            demands,
            self.batch_size,
            self.seq_len,
            self.pred_len,
        )

        train_data = self.data_loader.get_train()
        self.test_data = self.data_loader.get_test()

        self.model = build_model(
            input_shape=(self.seq_len, self.data_loader.n_feature),
            pred_len=self.pred_len,
            norm_type=self.norm_type,
            activation=self.activation,
            dropout=self.dropout,
            n_block=self.n_block,
            ff_dim=self.ff_dim,
            target_slice=self.data_loader.target_slice,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        self.model.fit(
          train_data,
          epochs=self.train_epochs,
          verbose=0
        )

    def forecast(self, demand_test, weather_test):
        forecasts = self.model.predict(self.test_data, verbose=0)
        tms_forecasts = pd.DataFrame(self.data_loader.inverse_transform(forecasts[-1,:,:].reshape(forecasts.shape[1], forecasts.shape[2])))
        return tms_forecasts.values


tsmix = {
    'name': 'TSMix',
    'model': TSMix(train_epochs=50, dropout=0.8),
    'preprocessing': {
        'demand': [Logarithm(), LGBM_impute_nan_demand()],
        'weather': []
    }
}
