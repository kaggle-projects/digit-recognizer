import datetime
import os
import sys
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Union

import pandas as pd
from sklearn import svm
from sklearn.exceptions import ConvergenceWarning
from utensil import get_logger
from utensil.random_search import RandomizedConfig, ExponentialBetweenParam, RandomizedChoices, UniformBetweenParam, \
    RandomizedDispatcher, RandomSearch, SeededConfig, ModelScore


script_name = os.path.basename(__file__)
logger = get_logger(script_name)


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
    timeout: float
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
    timeout=300,
    train_ratio=UniformBetweenParam(0.3, 0.6, float),

    c=ExponentialBetweenParam(1e-6, 1e3, float),
    kernel=RandomizedChoices(KernelChoices.POLY, KernelChoices.RBF, KernelChoices.SIGMOID),
    kernel_config=RandomizedDispatcher('kernel', {
        KernelChoices.POLY: poly_config_temp,
        KernelChoices.RBF: rbf_config_temp,
        KernelChoices.SIGMOID: sigmoid_config_temp
    })
)


# endregion


class KernelSvmRandomSearch(RandomSearch):
    def get_xy(self, tr_path, te_path):
        tr_path = os.path.normpath(tr_path)
        te_path = os.path.normpath(te_path)
        _x = pd.read_csv(tr_path).append(pd.read_csv(te_path))

        _y = _x['label']
        _x = _x.drop(columns='label')

        return _x, _y

    def model_scores_to_csv(self, model_scores):
        df = pd.DataFrame.from_dict(model_scores, orient='index')
        df.index = df.index.rename('model_id')
        df_columns = df.columns.tolist()
        df_columns[df_columns.index('score')], df_columns[0] = df_columns[0], df_columns[df_columns.index('score')]
        df = df[df_columns]
        df.to_csv(f'{script_name}.model_scores.csv')

    def do_training(self, sd_config: SeededConfig, train_x, train_y, idx) -> ModelScore:
        # get model config by model_id
        self.logger.info(f'model_id={sd_config.cid}: config={sd_config.config.to_plain_dict()}')

        # get validation set
        tr_size = int(train_x.shape[0] * sd_config.config.train_ratio)
        tr_x = train_x.iloc[idx[:tr_size]]
        tr_y = train_y.iloc[idx[:tr_size]]
        val_x = train_x.iloc[idx[tr_size:]]
        val_y = train_y.iloc[idx[tr_size:]]

        # prepare arguments for svc
        svc_kwargs = dict(
            C=sd_config.config.c,
            kernel=sd_config.config.kernel.value,
        )

        # prepare arguments of each kernel type for svc
        if isinstance(sd_config.config.kernel_config, RbfConfig):
            svc_kwargs.update(dict(
                gamma=sd_config.config.kernel_config.gamma
            ))
        elif isinstance(sd_config.config.kernel_config, PolyConfig):
            svc_kwargs.update(dict(
                gamma=sd_config.config.kernel_config.gamma,
                degree=sd_config.config.kernel_config.degree
            ))
        elif isinstance(sd_config.config.kernel_config, SigmoidConfig):
            svc_kwargs.update(dict(
                gamma=sd_config.config.kernel_config.gamma
            ))
        else:
            raise NotImplemented

        # train and validation
        self.logger.info(f'model_id={sd_config.cid}: training')
        model = svm.SVC(max_iter=1, **svc_kwargs)
        elapse = 0
        while elapse < sd_config.config.timeout:
            with warnings.catch_warnings(record=True) as wrn:
                warnings.simplefilter('always', category=ConvergenceWarning)
                st = datetime.datetime.now()
                model.fit(tr_x, tr_y)
                if not any([w.category is ConvergenceWarning for w in wrn]):
                    break
                elapse += (datetime.datetime.now() - st).total_seconds()
                sys.stdout.write('.')
                sys.stdout.flush()
        else:
            self.logger.info(f'model_id={sd_config.cid}: timeout, {elapse:3g} >= {sd_config.config.timeout:3g}')
        score = model.score(val_x, val_y)
        self.logger.info(f'model_id={sd_config.cid}: score={score}')
        return ModelScore(model, score)


def search():
    kernel_svm_rsearch = KernelSvmRandomSearch(logger)
    kernel_svm_rsearch.train('data/train.csv', 'data/test.csv', svm_config_temp)


if __name__ == '__main__':
    search()
