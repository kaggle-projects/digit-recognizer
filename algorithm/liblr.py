import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
from liblinear.liblinearutil import *
from utensil import get_logger
from utensil.random_search import RandomizedConfig, ExponentialBetweenParam, RandomizedChoices, UniformBetweenParam, \
    SeededConfig, RandomSearch, ModelScore

if __file__:
    import os
    script_name = os.path.basename(__file__)
else:
    script_name = __name__
logger = get_logger(script_name)


# regionConfig Definition


class SolverTypeChoices(Enum):
    L2R_LR = 0
    L2R_L2LOSS_SVC_DUAL = 1
    L2R_L2LOSS_SVC = 2
    L2R_L1LOSS_SVC_DUAL = 3
    MCSVM_CS = 4
    L1R_L2LOSS_SVC = 5
    L1R_LR = 6
    L2R_LR_DUAL = 7


@dataclass
class LinearSvmConfig(RandomizedConfig):
    train_ratio: Union[float, UniformBetweenParam]
    c: Union[float, ExponentialBetweenParam]
    solver_type: Union[SolverTypeChoices, RandomizedChoices]


liblr_config_temp = LinearSvmConfig(
    train_ratio=UniformBetweenParam(0.3, 0.6, float),

    c=ExponentialBetweenParam(1e-6, 1e3, float),
    solver_type=RandomizedChoices(
        SolverTypeChoices.L2R_LR, SolverTypeChoices.L2R_L2LOSS_SVC_DUAL,
        SolverTypeChoices.L2R_L2LOSS_SVC, SolverTypeChoices.L2R_L1LOSS_SVC_DUAL,
        SolverTypeChoices.L1R_L2LOSS_SVC,
        SolverTypeChoices.L1R_LR, SolverTypeChoices.L2R_LR_DUAL,
    )
)


# endregion


class LiblrModelWrapper:
    def __init__(self, liblr_model):
        self.liblr_model = liblr_model

    def predict(self, te_x):
        p_label, _, _ = predict([], te_x, self.liblr_model)
        return p_label

class LinearSvmRandomSearch(RandomSearch):
    MODEL_NAME = 'linearsvm'
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
        df.to_csv(f'{self.MODEL_NAME}.model_scores.csv')

    def do_training(self, sd_config: SeededConfig, train_x, train_y, idx):
        # get model config by model_id
        self.logger.info(f'model_id={sd_config.cid}: config={sd_config.config.to_plain_dict()}')

        # get validation set
        tr_size = int(train_x.shape[0] * sd_config.config.train_ratio)
        tr_x = train_x.iloc[idx[:tr_size]].astype(pd.SparseDtype(fill_value=0)).sparse.to_coo().tocsr()
        tr_y = train_y.iloc[idx[:tr_size]].to_numpy()
        val_x = train_x.iloc[idx[tr_size:]].astype(pd.SparseDtype(fill_value=0)).sparse.to_coo().tocsr()
        val_y = train_y.iloc[idx[tr_size:]].to_numpy()

        # prepare arguments for svc
        param = f'-c {sd_config.config.c} -s {sd_config.config.solver_type.value}'


        # train and validation
        self.logger.info(f'model_id={sd_config.cid}: training')
        model = train(tr_y, tr_x, param)
        # model.set_params(max_iter=max_iter)
        p_label, p_acc, p_val = predict(val_y, val_x, model)
        self.logger.info(f'model_id={sd_config.cid}: score={p_acc[0]}')
        return ModelScore(LiblrModelWrapper(model), p_acc[0])

    def train(self, tr_path, te_path, config_temp, model_id_range=None, seed=0):
        if model_id_range is None:
            model_id_range = range(10)

        x, y = self.get_xy(tr_path, te_path)

        te_x = x[y.isna()].astype(pd.SparseDtype(fill_value=0)).sparse.to_coo().tocsr()
        train_x = x[~y.isna()]
        train_y = y[~y.isna()]

        rng = np.random.default_rng(seed)
        idx = np.arange(train_x.shape[0])
        rng.shuffle(idx)

        if not os.path.exists('submit'):
            os.mkdir('submit')
        model_scores = {}
        best_model = None

        for mid in model_id_range:
            self.logger.info(f'model_id={mid}: initialize')

            sd_config = SeededConfig.from_config_template(config_temp=config_temp, model_id=mid, seed=seed)
            model, score = None, None
            try:
                model, score = self.do_training(sd_config, train_x, train_y, idx)
            except Exception:
                self.logger.warning(f'model_id={sd_config.cid}: invalid config for model_id={sd_config.cid}, seed={seed}',
                               stack_info=True)

            self.logger.info(f'model_id={sd_config.cid}: score={score}')

            # record model_id, model_config and score
            model_scores[sd_config.cid] = sd_config.config.to_plain_dict()
            assert 'score' not in model_scores[sd_config.cid]
            model_scores[sd_config.cid]['score'] = score
            self.model_scores_to_csv(model_scores)

            # keep the best model
            if score is not None and (best_model is None or best_model[0] < score):
                best_model = (score, model, sd_config)

                # use the current best model to generate csv for submission
                te_xpy = np.empty(shape=(te_x.shape[0], 2), dtype=int)
                te_xpy[:, 0] = np.arange(te_x.shape[0]) + 1
                te_xpy[:, 1] = best_model[1].predict(te_x)
                submit_path = os.path.join('submit', f'{__name__}_{int(np.round(score * 1e5))}_{sd_config.cid:04d}.csv')
                pd.DataFrame(te_xpy, columns=['ImageId', 'Label']).to_csv(submit_path, index=False)
            if best_model is not None:
                self.logger.info(
                    f'model_id={sd_config.cid}: current best model_id={best_model[2].cid}, score={best_model[0]}, '
                    f'config={best_model[2].config.to_plain_dict()}')

        return best_model, model_scores



def search():
    liblr_rsearch = LinearSvmRandomSearch(logger)
    liblr_rsearch.train('data/train.csv', 'data/test.csv', liblr_config_temp, model_id_range=range(109,1000))


if __name__ == '__main__':
    search()
