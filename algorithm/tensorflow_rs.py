import datetime
import hashlib
import os
from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.models import Sequential
from utensil import get_logger
from utensil.random_search import RandomizedConfig, ExponentialBetweenParam, RandomizedChoices, UniformBetweenParam, \
    BooleanParam

if __file__:
    script_name = os.path.basename(__file__)
else:
    script_name = __name__
logger = get_logger(script_name)


def csv_to_tf_image_directory(csv_name, shape, out_root, has_label):
    tr_data = np.genfromtxt(csv_name, delimiter=',', skip_header=1)
    for i in range(tr_data.shape[0]):
        if has_label:
            img_dir = os.path.join(out_root, f'{int(tr_data[i, 0])}')
            im = Image.fromarray(tr_data[i, 1:].reshape(shape))
        else:
            img_dir = os.path.join(out_root, '0')
            im = Image.fromarray(tr_data[i, :].reshape(shape))
        im = im.convert('RGB')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        out_fname = os.path.join(img_dir, f'{i+1:05d}.png')
        im.save(out_fname)

class EmptyLayer:
    pass

class RescalingChoices(Enum):
    NO_RESCALE = 0
    ZERO_ONE = 1
    NEG_ONE_POS_ONE = 2

class PoolingChoices(Enum):
    NO_POOLING = 0
    MAX = 1
    AVG = 2

class PoolingPositionChoices(Enum):
    BEFORE_CONV = 0
    BEFORE_DROPOUT = 1
    AFTER_DROPOUT = 2


class PaddingChoices(Enum):
    VALID = 0
    SAME = 1


class ActivationChoices(Enum):
    RELU = 0
    SIGMOID = 1

class OptimizerChoices(Enum):
    ADADELTA = 0
    ADAGRAD = 1
    ADAM = 2
    ADAMAX = 3
    FTRL = 4
    NADAM = 5
    RMSPROP = 6
    SGD = 7

@dataclass
class CnnConfig(RandomizedConfig):
    base_seed: int
    rescaling: Union[RescalingChoices, RandomizedChoices]
    pool_method: Union[PoolingChoices, RandomizedChoices]
    pool_size: Union[int, UniformBetweenParam]
    pool_padding: Union[PaddingChoices, RandomizedChoices]
    pool_position: Union[PoolingPositionChoices, RandomizedChoices]

    conv2d_layers: Union[int, UniformBetweenParam]
    conv2d_padding: Union[PaddingChoices, RandomizedChoices]
    conv2d_act: Union[ActivationChoices, RandomizedChoices]
    conv2d_kernel_size: Union[int, UniformBetweenParam]
    conv2d_filters_start: Union[int, ExponentialBetweenParam]

    dropout_do: Union[bool, BooleanParam]
    dropout_rate: Union[float, UniformBetweenParam]

    dense_layers: Union[int, UniformBetweenParam]
    dense_units_start: Union[int, UniformBetweenParam]
    dense_act: Union[ActivationChoices, RandomizedChoices]

    optimizer: Union[OptimizerChoices, RandomizedChoices]
    fit_patience: Union[int, UniformBetweenParam]
    fit_min_delta: Union[float, ExponentialBetweenParam]
    fit_timeout: float

@dataclass
class DataConfig(RandomizedConfig):
    base_seed: int
    train_data_dir: str
    test_data_dir: str

    data_min: float
    data_max: float
    num_classes: int
    img_size: Union[int, UniformBetweenParam]
    train_ratio: Union[float, UniformBetweenParam]

def choose_rescaling(rescaling_choice, data_min, data_max):
    if rescaling_choice is RescalingChoices.NO_RESCALE:
        return EmptyLayer
    elif rescaling_choice is RescalingChoices.ZERO_ONE:
        return tf.keras.layers.experimental.preprocessing.Rescaling(
            scale=1. / (data_max - data_min),
            offset=-data_min / (data_max - data_min)
        )
    elif rescaling_choice is RescalingChoices.NEG_ONE_POS_ONE:
        return tf.keras.layers.experimental.preprocessing.Rescaling(
            scale=2. / (data_max - data_min),
            offset=-(data_max + data_min) / (data_max - data_min)
        )
    else:
        raise NotImplementedError
    
def choose_conv2d_layer(conv2d_filters_start, conv2d_i, kernel_size, padding, activation):
    return tf.keras.layers.Conv2D(
        filters=conv2d_filters_start * 2**conv2d_i,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation
    )

def choose_dropout_layer(dropout_do, dropout_rate):
    if dropout_do:
        return tf.keras.layers.Dropout(dropout_rate)
    else:
        return EmptyLayer
    
def choose_pooling_layer(pooling_choice, pool_size, padding):
    if pooling_choice is PoolingChoices.NO_POOLING:
        return EmptyLayer
    elif pooling_choice is PoolingChoices.MAX:
        return tf.keras.layers.MaxPool2D(
            pool_size=(pool_size, pool_size),
            padding=padding,
        )
    elif pooling_choice is PoolingChoices.AVG:
        return tf.keras.layers.AveragePooling2D(
            pool_size=(pool_size, pool_size),
            padding=padding,
        )
    else:
        raise NotImplementedError

def choose_dense_layer(num_classes, dense_units_start, dense_i, dense_activation):
    return tf.keras.layers.Dense(
        units=max(num_classes, int(dense_units_start / 2 ** dense_i)),
        activation=dense_activation
    )
    
def choose_pooling_position(pooling_position_choice, conv2d_layer, pool_layer, dropout_layer):
    if pooling_position_choice is PoolingPositionChoices.BEFORE_CONV:
        return pool_layer, conv2d_layer, dropout_layer
    elif pooling_position_choice is PoolingPositionChoices.BEFORE_DROPOUT:
        return conv2d_layer, pool_layer, dropout_layer
    elif pooling_position_choice is PoolingPositionChoices.AFTER_DROPOUT:
        return conv2d_layer, dropout_layer, pool_layer
    else:
        raise NotImplementedError

def choose_padding(padding_choice):
    if padding_choice is PaddingChoices.VALID:
        return 'valid'
    elif padding_choice is PaddingChoices.SAME:
        return 'same'
    else:
        raise NotImplementedError
    
def choose_activation(activation_choice):
    if activation_choice is ActivationChoices.RELU:
        return 'relu'
    elif activation_choice is ActivationChoices.SIGMOID:
        return 'sigmoid'
    else:
        raise NotImplementedError
    
def choose_optimizer(optimizer_choice):
    if optimizer_choice is OptimizerChoices.ADADELTA:
        return 'adadelta'
    elif optimizer_choice is OptimizerChoices.ADAGRAD:
        return 'adagrad'
    elif optimizer_choice is OptimizerChoices.ADAM:
        return 'adam'
    elif optimizer_choice is OptimizerChoices.ADAMAX:
        return 'adamax'
    elif optimizer_choice is OptimizerChoices.FTRL:
        return 'ftrl'
    elif optimizer_choice is OptimizerChoices.NADAM:
        return 'nadam'
    elif optimizer_choice is OptimizerChoices.RMSPROP:
        return 'rmsprop'
    elif optimizer_choice is OptimizerChoices.SGD:
        return 'sgd'
    else:
        raise NotImplementedError

def get_train_dataset(train_data_dir, train_ratio, img_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        validation_split=1-train_ratio,
        subset="training",
        seed=0,
        image_size=(img_size, img_size),
        batch_size=32
    )
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds

def get_val_dataset(train_data_dir, train_ratio, img_size):
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        validation_split=1-train_ratio,
        subset="validation",
        seed=0,
        image_size=(img_size, img_size),
        batch_size=32
    )
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return val_ds

def get_test_dataset(test_data_dir, img_size):
    return tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode=None,
        image_size=(img_size, img_size),
        batch_size=32,
        shuffle=False
    )
    
def build_model_layers(data_config: DataConfig, cnn_config: CnnConfig):
    model_layers = [
        tf.keras.layers.InputLayer(input_shape=(data_config.img_size, data_config.img_size, 3)),
        choose_rescaling(cnn_config.rescaling, data_config.data_min, data_config.data_max)
    ]
    pool_padding = choose_padding(cnn_config.pool_padding)
    conv2d_padding = choose_padding(cnn_config.conv2d_padding)
    conv2d_activation = choose_activation(cnn_config.conv2d_act)

    for conv2d_i in range(cnn_config.conv2d_layers):
        pool_layer = choose_pooling_layer(cnn_config.pool_method, cnn_config.pool_size, pool_padding)
        conv2d_layer = choose_conv2d_layer(cnn_config.conv2d_filters_start,
                                           conv2d_i,
                                           cnn_config.conv2d_kernel_size,
                                           conv2d_padding,
                                           conv2d_activation)
        dropout_layer = choose_dropout_layer(cnn_config.dropout_do, cnn_config.dropout_rate)

        model_layers.extend(choose_pooling_position(cnn_config.pool_position, conv2d_layer, pool_layer, dropout_layer))

    model_layers.append(tf.keras.layers.Flatten())
    dense_activation = choose_activation(cnn_config.dense_act)

    for dense_i in range(cnn_config.dense_layers):
        model_layers.append(
            choose_dense_layer(data_config.num_classes, cnn_config.dense_units_start, dense_i, dense_activation))

    model_layers.append(tf.keras.layers.Dense(data_config.num_classes))
    return model_layers

def build_callbacks(checkpoint_filepath, fit_min_delta, fit_patience, fit_timeout):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=fit_min_delta,
            patience=fit_patience,
            verbose=1,
        ),
        TimeoutCallback(fit_timeout)
    ]

class TimeoutCallback(tf.keras.callbacks.Callback):
    def __init__(self, timeout):
        super().__init__()
        self.started = False
        self.st = None
        self.timeout = timeout

    def on_train_begin(self, logs=None):
        if not self.started:
            self.st = datetime.datetime.now()
            self.started = True

    def on_epoch_end(self, epoch, logs=None):
        ed = datetime.datetime.now()
        if (ed - self.st).total_seconds() > self.timeout:
            self.model.stop_training = True

def main():
    # csv_to_tf_image_directory('data/train.csv', (28,28), 'data/images/train', True)
    # csv_to_tf_image_directory('data/test.csv', (28,28), 'data/images/test', False)

    cnn_config_tmp = CnnConfig(
        base_seed=1,
    
        rescaling=RandomizedChoices(RescalingChoices.NO_RESCALE,
                                    RescalingChoices.ZERO_ONE,
                                    RescalingChoices.NEG_ONE_POS_ONE),
        pool_method=RandomizedChoices(PoolingChoices.NO_POOLING,
                                  PoolingChoices.MAX,
                                  PoolingChoices.AVG),
        pool_size=UniformBetweenParam(2, 5, int),
        pool_padding=RandomizedChoices(PaddingChoices.SAME, PaddingChoices.VALID),
        pool_position=RandomizedChoices(PoolingPositionChoices.BEFORE_CONV,
                                        PoolingPositionChoices.BEFORE_DROPOUT,
                                        PoolingPositionChoices.AFTER_DROPOUT),
    
        conv2d_layers=UniformBetweenParam(3, 6, int),
        conv2d_padding=RandomizedChoices(PaddingChoices.SAME, PaddingChoices.VALID),
        conv2d_act=RandomizedChoices(ActivationChoices.RELU, ActivationChoices.SIGMOID),
        conv2d_kernel_size=UniformBetweenParam(2, 8, int),
        conv2d_filters_start=ExponentialBetweenParam(8, 64, int),
    
        dropout_do=BooleanParam(),
        dropout_rate=UniformBetweenParam(0, 0.5, float),
    
        dense_layers=UniformBetweenParam(1, 5, int),
        dense_units_start=UniformBetweenParam(32, 256, int),
        dense_act=RandomizedChoices(ActivationChoices.RELU, ActivationChoices.SIGMOID),
    
        optimizer=RandomizedChoices(
            OptimizerChoices.ADADELTA,
            OptimizerChoices.ADAGRAD,
            OptimizerChoices.ADAM,
            OptimizerChoices.ADAMAX,
            OptimizerChoices.FTRL,
            OptimizerChoices.NADAM,
            OptimizerChoices.RMSPROP,
            OptimizerChoices.SGD,
        ),
        fit_patience=UniformBetweenParam(2, 32, int),
        fit_min_delta=ExponentialBetweenParam(1e-5, 1e-1, float),
        fit_timeout=300,
    )
    
    data_config_tmp = DataConfig(
        base_seed=1,
        train_data_dir='data/images/train',
        test_data_dir='data/images/test',
        data_min=0,
        data_max=255,
        num_classes=10,
        img_size=UniformBetweenParam(14, 29, int),
        train_ratio=UniformBetweenParam(0.3, 0.7, float),
    )
    
    pred_path = os.path.join(
        'data', 'prediction',
        hashlib.sha256(str((sorted(data_config_tmp.to_plain_dict().items()), sorted(cnn_config_tmp.to_plain_dict().items()))).encode()).hexdigest()[:8]
    )
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    df_score = pd.DataFrame(
        [], columns=['cnn_config_i', 'score', 'data_config_i'] +
                    list(cnn_config_tmp.to_plain_dict().keys())+
                    list(data_config_tmp.to_plain_dict().keys())
    )
    
    checkpoint_filepath = 'data/cache/checkpoint'
    cnn_config_range = iter(range(10000000))
    data_config_range = iter(range(10000000))
    cnt_bad_setting = 0
    cnt_ttl_setting = 0
    best_score = [[], None]
    
    for _ in range(1000):
        data_config_i = next(data_config_range)
        data_config_r, data_config = data_config_tmp.get_config(data_config_i, seed=data_config_tmp.base_seed)
    
        train_ds = get_train_dataset(data_config.train_data_dir, data_config.train_ratio, data_config.img_size)
        val_ds = get_val_dataset(data_config.train_data_dir, data_config.train_ratio, data_config.img_size)
        test_ds = get_test_dataset(data_config.test_data_dir, data_config.img_size)
    
        for _ in range(10):
            cnn_config_i = next(cnn_config_range)
            cnn_config_r, cnn_config = cnn_config_tmp.get_config(cnn_config_i, seed=cnn_config_tmp.base_seed)
            cnt_ttl_setting += 1
            try:
                model_layers = build_model_layers(data_config, cnn_config)
                model = Sequential([model_layer for model_layer in model_layers if model_layer is not EmptyLayer])
    
                optimizer = choose_optimizer(cnn_config.optimizer)
    
                model.compile(
                    optimizer=optimizer,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'],
                )
    
                model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=1000,
                    callbacks=build_callbacks(checkpoint_filepath,
                                              cnn_config.fit_min_delta,
                                              cnn_config.fit_patience,
                                              cnn_config.fit_timeout),
                )
    
                model.load_weights(checkpoint_filepath)
                loss, acc = model.evaluate(val_ds)
                score = acc
    
            except Exception as e:
                cnt_bad_setting += 1
                logger.warning(f'invalid setting ({cnt_bad_setting}/{cnt_ttl_setting})')
                score = np.nan
                model = None
    
            if len(best_score[0]) == 0 or model is not None:
                best_score[0].append(score)
                best_score[0] = sorted(best_score[0])[::-1]
                best_score[1] = (data_config_i, cnn_config_i)
    
                if best_score[0][len(best_score[0])//10] <= score:
                    prediction = model.predict(test_ds)
                    pd.DataFrame.from_records(
                        list(zip(np.arange(prediction.shape[0]) + 1, prediction.argmax(axis=1))),
                        columns=['ImageId', 'Label']
                    ).to_csv(os.path.join(pred_path, f'{score:g}_{data_config_i:04d}_{cnn_config_i:05d}.csv'), index=False)
    
            logger.info(f'current best: score={best_score[0][0]:g}, id={best_score[1]}')
    
            cnn_config_dict = cnn_config.to_plain_dict()
            data_config_dict = data_config.to_plain_dict()
            s = pd.Series([], dtype=object)
            df_score = df_score.append(s, ignore_index=True)
            df_score.iloc[-1]['cnn_config_i'] = cnn_config_i
            df_score.iloc[-1]['data_config_i'] = data_config_i
            df_score.iloc[-1]['score'] = score
            for k, v in cnn_config_dict.items():
                df_score.iloc[-1][k] = v
            for k, v in data_config_dict.items():
                df_score.iloc[-1][k] = v
    
            df_score.to_csv('df_score.csv')

if __name__ == '__main__':
    main()
