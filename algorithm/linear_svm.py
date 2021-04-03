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
    BooleanParam, SeededConfig, RandomSearch, ModelScore


script_name = os.path.basename(__file__)
logger = get_logger(script_name)


# regionConfig Definition


class PenaltyChoices(Enum):
    L1 = 'l1'
    L2 = 'l2'


class LossChoices(Enum):
    HINGE = 'hinge'
    SQUARED_HINGE = 'squared_hinge'


@dataclass
class LinearSvmConfig(RandomizedConfig):
    timeout: float
    train_ratio: Union[float, UniformBetweenParam]
    c: Union[float, ExponentialBetweenParam]
    dual: Union[bool, BooleanParam]
    penalty: Union[PenaltyChoices, RandomizedChoices]
    loss: Union[LossChoices, RandomizedChoices]


linear_svm_config_temp = LinearSvmConfig(
    timeout=300,
    train_ratio=UniformBetweenParam(0.3, 0.6, float),

    c=ExponentialBetweenParam(1e-6, 1e3, float),
    dual=BooleanParam(),
    penalty=RandomizedChoices(PenaltyChoices.L1, PenaltyChoices.L2),
    loss=RandomizedChoices(LossChoices.HINGE, LossChoices.SQUARED_HINGE),
)


# endregion


class LinearSvmRandomSearch(RandomSearch):

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

    def do_training(self, sd_config: SeededConfig, train_x, train_y, idx):
        # get model config by model_id
        logger.info(f'model_id={sd_config.cid}: config={sd_config.config.to_plain_dict()}')

        # get validation set
        tr_size = int(train_x.shape[0] * sd_config.config.train_ratio)
        tr_x = train_x.iloc[idx[:tr_size]]
        tr_y = train_y.iloc[idx[:tr_size]]
        val_x = train_x.iloc[idx[tr_size:]]
        val_y = train_y.iloc[idx[tr_size:]]

        # prepare arguments for svc
        linear_svc_kwargs = dict(
            C=sd_config.config.c,
            penalty=sd_config.config.penalty.value,
            loss=sd_config.config.loss.value,
            dual=sd_config.config.dual,
        )

        # train and validation
        logger.info(f'model_id={sd_config.cid}: training')
        model = svm.LinearSVC(max_iter=1, **linear_svc_kwargs)
        elapse = 0
        while elapse < sd_config.config.timeout:
            with warnings.catch_warnings(record=True) as wrn:
                warnings.simplefilter('always', category=ConvergenceWarning)
                st = datetime.datetime.now()
                model.fit(tr_x, tr_y)
                if not any([w.category is ConvergenceWarning for w in wrn]):
                    break
                elapse += (datetime.datetime.now() - st).total_seconds()
                sys.stdout.write('..')
                sys.stdout.flush()
        else:
            logger.info(f'model_id={sd_config.cid}: timeout, {elapse:3g} >= {sd_config.config.timeout:3g}')
        score = model.score(val_x, val_y)
        logger.info(f'model_id={sd_config.cid}: score={score}')
        return ModelScore(model, score)


def search():
    linear_svm_rsearch = LinearSvmRandomSearch(logger)
    linear_svm_rsearch.train('data/train.csv', 'data/test.csv', linear_svm_config_temp, model_id_range=range(1000))


if __name__ == '__main__':
    search()
