import os
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
from sklearn import svm
from utensil import get_logger
from utensil.random_search import RandomizedConfig, ExponentialBetweenParam, RandomizedChoices, UniformBetweenParam, \
    RandomizedDispatcher

logger = get_logger(__name__)


# regionConfig Definition


class KernelChoices(Enum):
    POLY = 'poly'
    RBF = 'rbf'
    SIGMOID = 'sigmoid'


@dataclass
class PolyConfig(RandomizedConfig):
    gamma: Union[float, ExponentialBetweenParam]
    # coef0: Union[float, ExponentialBetweenParam]
    degree: Union[int, UniformBetweenParam]


@dataclass
class RbfConfig(RandomizedConfig):
    gamma: Union[float, ExponentialBetweenParam]


@dataclass
class SigmoidConfig(RandomizedConfig):
    gamma: Union[float, ExponentialBetweenParam]


@dataclass
class SvmConfig(RandomizedConfig):
    train_ratio: Union[float, UniformBetweenParam]
    c: Union[float, ExponentialBetweenParam]
    kernel: Union[Enum, RandomizedChoices]
    kernel_config: RandomizedDispatcher


poly_config_temp = PolyConfig(
    gamma=ExponentialBetweenParam(1e-6, 1e3, float),
    degree=UniformBetweenParam(2, 10, int),
)

rbf_config_temp = RbfConfig(
    gamma=ExponentialBetweenParam(1e-6, 1e3, float),
)

sigmoid_config_temp = SigmoidConfig(
    gamma=ExponentialBetweenParam(1e-6, 1e3, float),
)

svm_config_temp = SvmConfig(
    train_ratio=UniformBetweenParam(0.3, 0.6, float),

    c=ExponentialBetweenParam(1e-6, 1e3, float),
    kernel=RandomizedChoices(KernelChoices),
    kernel_config=RandomizedDispatcher('kernel', {
        KernelChoices.POLY: poly_config_temp,
        KernelChoices.RBF: rbf_config_temp,
        KernelChoices.SIGMOID: sigmoid_config_temp
    })
)


# endregion


def get_xy(tr_path, te_path):
    tr_path = os.path.normpath(tr_path)
    te_path = os.path.normpath(te_path)
    _x = pd.read_csv(tr_path).append(pd.read_csv(te_path))

    _y = _x['label']
    _x = _x.drop(columns='label')

    return _x, _y


def model_scores_to_csv(_model_scores):
    df = pd.DataFrame.from_dict(_model_scores, orient='index')
    df.index = df.index.rename('model_id')
    df_columns = df.columns.tolist()
    df_columns[df_columns.index('score')], df_columns[0] = df_columns[0], df_columns[df_columns.index('score')]
    df = df[df_columns]
    df.to_csv(f'{__name__}.model_scores.csv')


def train(tr_path, te_path, model_id_range=None):
    if model_id_range is None:
        model_id_range = range(10)

    x, y = get_xy(tr_path, te_path)

    te_x = x[y.isna()]
    train_x = x[~y.isna()]
    train_y = y[~y.isna()]

    rng = np.random.default_rng(0)
    idx = np.arange(train_x.shape[0])
    rng.shuffle(idx)

    if not os.path.exists('submit'):
        os.mkdir('submit')
    model_scores = {}
    best_model = None

    for model_id in model_id_range:
        logger.info(f'{model_id=}: initialize')

        # get model config by model_id
        model_r, svm_config = svm_config_temp.get_config(model_id)
        logger.info(f'{model_id=}: config={svm_config.to_plain_dict()}')

        # get validation set
        tr_size = int(train_x.shape[0] * svm_config.train_ratio)
        tr_x = train_x.iloc[idx[:tr_size]]
        tr_y = train_y.iloc[idx[:tr_size]]
        val_x = train_x.iloc[idx[tr_size:]]
        val_y = train_y.iloc[idx[tr_size:]]

        # prepare arguments for svc
        svc_kwargs = dict(
            C=svm_config.c,
            kernel=svm_config.kernel.value,
        )

        # prepare arguments of each kernel type for svc
        if isinstance(svm_config.kernel_config, RbfConfig):
            svc_kwargs.update(dict(
                gamma=svm_config.kernel_config.gamma
            ))
        elif isinstance(svm_config.kernel_config, PolyConfig):
            svc_kwargs.update(dict(
                gamma=svm_config.kernel_config.gamma,
                degree=svm_config.kernel_config.degree
            ))
        else:
            raise NotImplemented

        # train and validation
        logger.info(f'{model_id=}: training')
        model = svm.SVC(**svc_kwargs)
        model.fit(tr_x, tr_y)
        score = model.score(val_x, val_y)
        logger.info(f'{model_id=}: {score=}')

        # record model_id, model_config and score
        model_scores[model_id] = svm_config.to_plain_dict()
        assert 'score' not in model_scores[model_id]
        model_scores[model_id]['score'] = score
        model_scores_to_csv(model_scores)

        # keep the best model
        if best_model is None or best_model[0] < score:
            best_model = (score, deepcopy(model), model_id, svm_config)

            # use the current best model to generate csv for submission
            te_xpy = np.empty(shape=(te_x.shape[0], 2), dtype=int)
            te_xpy[:, 0] = np.arange(te_x.shape[0]) + 1
            te_xpy[:, 1] = best_model[1].predict(te_x)
            submit_path = os.path.join('submit', f'{__name__}_{int(np.round(score * 1e5))}_{model_id:04d}.csv')
            pd.DataFrame(te_xpy, columns=['ImageId', 'Label']).to_csv(submit_path, index=False)

        logger.info(
            f'{model_id=}: current best model_id={best_model[2]}, score={best_model[0]}, config={svm_config.to_plain_dict()}')
